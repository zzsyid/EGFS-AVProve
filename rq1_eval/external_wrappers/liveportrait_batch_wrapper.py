#!/usr/bin/env python3
"""Run multiple LivePortrait reenactment tasks in one long-lived process.

This is a throughput wrapper for M6-style task CSVs. The regular
liveportrait_reenactment_wrapper.py launches LivePortrait's inference.py once per
row, which reloads model weights and repeats face/landmark warmup every time.
This batch wrapper constructs LivePortraitPipeline once, then calls execute()
for each selected row.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REQUIRED_COLUMNS = {
    "task_id",
    "generator",
    "input_video_path",
    "donor_video_path",
    "output_video_path",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    rows = list(rows)
    if not rows:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(command: List[str], dry_run: bool = False) -> None:
    print("[liveportrait-batch]", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def runtime_env(liveportrait_root: Path, ffmpeg: str) -> Dict[str, str]:
    env = os.environ.copy()
    candidates = [
        "/path/to/runtime/anaconda3/pkgs/libcublas-11.11.3.6-0/lib",
        "/path/to/runtime/anaconda3/pkgs/libcufft-10.9.0.58-0/lib",
        "/path/to/runtime/anaconda3/pkgs/libcurand-10.3.4.101-0/lib",
        "/path/to/runtime/anaconda3/envs/cxr/lib/python3.11/site-packages/nvidia/cuda_runtime/lib",
        str(Path(sys.prefix) / "lib/python3.10/site-packages/nvidia/cudnn/lib"),
    ]
    existing = [item for item in candidates if Path(item).exists()]
    if existing:
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(existing + ([current] if current else []))
    ffmpeg_dir = str(Path(ffmpeg).resolve().parent)
    env["PATH"] = os.pathsep.join([ffmpeg_dir, str(liveportrait_root / "ffmpeg"), env.get("PATH", "")])
    return env


def ensure_runtime_env(root: Path, ffmpeg: str) -> None:
    env = runtime_env(root, ffmpeg)
    if os.environ.get("RQ1_LIVEPORTRAIT_BATCH_ENV_READY") != "1":
        env["RQ1_LIVEPORTRAIT_BATCH_ENV_READY"] = "1"
        os.execve(sys.executable, [sys.executable] + sys.argv, env)
    os.environ.update(env)


def validate_install(root: Path) -> None:
    required = [
        "inference.py",
        "src/live_portrait_pipeline.py",
        "pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth",
        "pretrained_weights/liveportrait/base_models/motion_extractor.pth",
        "pretrained_weights/liveportrait/base_models/spade_generator.pth",
        "pretrained_weights/liveportrait/base_models/warping_module.pth",
        "pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth",
    ]
    missing = [relative for relative in required if not (root / relative).exists()]
    if missing:
        raise FileNotFoundError("incomplete LivePortrait install:\n  - " + "\n  - ".join(missing))


def extract_source_frame(ffmpeg: str, input_video: Path, source_time: str, source_image: Path,
                         dry_run: bool) -> None:
    run([
        ffmpeg, "-y", "-ss", source_time, "-i", str(input_video),
        "-vf", "crop='min(iw,ih)':'min(iw,ih)',scale=512:512",
        "-frames:v", "1", str(source_image),
    ], dry_run=dry_run)


def remux_donor_audio(ffmpeg: str, generated: Path, donor_video: Path, output_video: Path,
                      dry_run: bool) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    run([
        ffmpeg, "-y", "-i", str(generated), "-i", str(donor_video),
        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac",
        "-shortest", str(output_video),
    ], dry_run=dry_run)


def partial_fields(target_class, kwargs):
    return target_class(**{key: val for key, val in kwargs.items() if hasattr(target_class, key)})


def import_liveportrait(root: Path):
    sys.path.insert(0, str(root))
    from src.config.argument_config import ArgumentConfig
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_pipeline import LivePortraitPipeline

    return ArgumentConfig, InferenceConfig, CropConfig, LivePortraitPipeline


def build_pipeline(root: Path, device_id: int):
    ArgumentConfig, InferenceConfig, CropConfig, LivePortraitPipeline = import_liveportrait(root)
    base_args = ArgumentConfig(
        flag_crop_driving_video=True,
        device_id=device_id,
    )
    inference_cfg = partial_fields(InferenceConfig, base_args.__dict__)
    crop_cfg = partial_fields(CropConfig, base_args.__dict__)
    pipeline = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
    return ArgumentConfig, pipeline


def process_row(row: Dict[str, str], args, ArgumentConfig, pipeline) -> None:
    input_video = Path(row["input_video_path"])
    donor_video = Path(row["donor_video_path"])
    output_video = Path(row["output_video_path"])
    for path in (input_video, donor_video):
        if not path.exists():
            raise FileNotFoundError(path)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    source_times = [item.strip() for item in args.source_times.split(",") if item.strip()]
    if not source_times:
        raise ValueError("--source-times must contain at least one timestamp")

    token = uuid.uuid4().hex[:12]
    last_exc: Optional[Exception] = None
    with tempfile.TemporaryDirectory(prefix="rq1_liveportrait_batch_") as raw_tmp:
        tmp = Path(raw_tmp)
        for attempt, source_time in enumerate(source_times, start=1):
            source_image = tmp / f"source_{token}_{attempt}.png"
            output_dir = tmp / f"animations_{attempt}"
            try:
                extract_source_frame(args.ffmpeg, input_video, source_time, source_image, args.dry_run)
                if args.dry_run:
                    return
                lp_args = ArgumentConfig(
                    source=str(source_image),
                    driving=str(donor_video),
                    output_dir=str(output_dir),
                    flag_crop_driving_video=True,
                    device_id=args.device_id,
                )
                generated, _concat = pipeline.execute(lp_args)
                remux_donor_audio(args.ffmpeg, Path(generated), donor_video, output_video, args.dry_run)
                if not output_video.exists():
                    raise FileNotFoundError(f"LivePortrait did not create {output_video}")
                return
            except Exception as exc:  # retry on alternate source frames
                last_exc = exc
                if attempt < len(source_times):
                    print(
                        f"[liveportrait-batch] attempt {attempt} failed for {row['task_id']}; "
                        f"retry source frame at {source_times[attempt]} sec: {exc}",
                        flush=True,
                    )
                    continue
                raise
    if last_exc:
        raise last_exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--generator", default="LivePortrait")
    parser.add_argument("--liveportrait-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-times", default="0,1,2,3")
    parser.add_argument("--failures-csv", type=Path, required=True)
    args = parser.parse_args()

    tasks_path = args.tasks.resolve()
    root = args.liveportrait_root.resolve()
    if args.device_id is None:
        if ":" in args.device:
            args.device_id = int(args.device.rsplit(":", 1)[1])
        else:
            args.device_id = 0
    ensure_runtime_env(root, args.ffmpeg)
    validate_install(root)

    rows = read_csv(tasks_path)
    missing = REQUIRED_COLUMNS - set(rows[0]) if rows else REQUIRED_COLUMNS
    if missing:
        raise ValueError(f"task CSV missing columns: {sorted(missing)}")
    rows = [row for row in rows if row["generator"] == args.generator]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < --num-shards")
    rows = rows[args.shard_index::args.num_shards]
    if args.limit:
        rows = rows[:args.limit]

    old_cwd = Path.cwd()
    os.chdir(root)
    try:
        ArgumentConfig, pipeline = build_pipeline(root, args.device_id)
        failures = []
        n_ok = n_skipped = 0
        for index, row in enumerate(rows, start=1):
            output = Path(row["output_video_path"])
            if args.resume and output.exists():
                n_skipped += 1
                print(f"[liveportrait-batch {index}/{len(rows)}] skip   {row['task_id']}", flush=True)
                continue
            try:
                process_row(row, args, ArgumentConfig, pipeline)
                n_ok += 1
                print(f"[liveportrait-batch {index}/{len(rows)}] ok     {row['task_id']}", flush=True)
            except Exception as exc:
                fail = dict(row)
                fail["error"] = str(exc)[-1000:]
                failures.append(fail)
                print(f"[liveportrait-batch {index}/{len(rows)}] FAILED {row['task_id']}: {exc}", flush=True)
        write_csv(args.failures_csv, failures)
        print(f"[liveportrait-batch] done ok={n_ok} skipped={n_skipped} failed={len(failures)}")
        if failures:
            raise SystemExit(1)
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
