#!/usr/bin/env python3
"""Run multiple AniPortrait tasks in one long-lived audio2vid invocation.

The per-row wrapper launches ``scripts.audio2vid`` once for every task, which
reloads Stable Diffusion, Wav2Vec, audio2mesh, and audio2pose weights every row.
AniPortrait's CLI already supports multiple ``test_cases`` in one YAML config,
so this wrapper builds one shard-level config, runs audio2vid once, and then
post-processes each generated grid video into the expected final output path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REQUIRED_COLUMNS = {
    "task_id",
    "generator",
    "input_video_path",
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


def run(command: List[str], cwd: Optional[Path] = None, dry_run: bool = False,
        env: Optional[dict] = None) -> None:
    print("[aniportrait-batch]", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=str(cwd) if cwd else None, check=True, env=env)


def yaml_string(path: Path) -> str:
    return json.dumps(str(path))


def device_visible_id(device: str) -> Optional[str]:
    if not device or device == "cpu":
        return None
    if device.startswith("cuda:"):
        return device.rsplit(":", 1)[1]
    if device == "cuda":
        return "0"
    raise ValueError(f"unsupported AniPortrait device: {device}")


def runtime_env(device: str) -> Dict[str, str]:
    env = os.environ.copy()
    visible_id = device_visible_id(device)
    if visible_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible_id
        print(
            f"[aniportrait-batch] binding child processes to CUDA_VISIBLE_DEVICES={visible_id}",
            flush=True,
        )
    return env


def validate_install(root: Path, accelerate: bool) -> None:
    required = [
        "scripts/audio2vid.py",
        "configs/inference/inference_audio.yaml",
        "configs/inference/inference_v2.yaml",
        "pretrained_model/stable-diffusion-v1-5",
        "pretrained_model/sd-vae-ft-mse",
        "pretrained_model/image_encoder",
        "pretrained_model/wav2vec2-base-960h",
        "pretrained_model/denoising_unet.pth",
        "pretrained_model/reference_unet.pth",
        "pretrained_model/pose_guider.pth",
        "pretrained_model/motion_module.pth",
        "pretrained_model/audio2mesh.pt",
        "pretrained_model/audio2pose.pt",
    ]
    if accelerate:
        required.append("pretrained_model/film_net_fp16.pt")
    missing = [relative for relative in required if not (root / relative).exists()]
    if missing:
        raise FileNotFoundError("incomplete AniPortrait install:\n  - " + "\n  - ".join(missing))


def write_audio_config(path: Path, root: Path, test_cases: List[Tuple[Path, Path]]) -> None:
    lines = [
        f"pretrained_base_model_path: {yaml_string(root / 'pretrained_model/stable-diffusion-v1-5')}",
        f"pretrained_vae_path: {yaml_string(root / 'pretrained_model/sd-vae-ft-mse')}",
        f"image_encoder_path: {yaml_string(root / 'pretrained_model/image_encoder')}",
        f"denoising_unet_path: {yaml_string(root / 'pretrained_model/denoising_unet.pth')}",
        f"reference_unet_path: {yaml_string(root / 'pretrained_model/reference_unet.pth')}",
        f"pose_guider_path: {yaml_string(root / 'pretrained_model/pose_guider.pth')}",
        f"motion_module_path: {yaml_string(root / 'pretrained_model/motion_module.pth')}",
        f"audio_inference_config: {yaml_string(root / 'configs/inference/inference_audio.yaml')}",
        f"inference_config: {yaml_string(root / 'configs/inference/inference_v2.yaml')}",
        'weight_dtype: "fp16"',
        "test_cases:",
    ]
    for source_image, driving_audio in test_cases:
        lines.append(f"  {yaml_string(source_image)}:")
        lines.append(f"    - {yaml_string(driving_audio)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_generated_video(root: Path, source_stem: str, audio_stem: str) -> Path:
    pattern = f"{source_stem}_{audio_stem}_*.mp4"
    candidates = [
        path for path in (root / "output").rglob(pattern)
        if not path.name.endswith("_noaudio.mp4")
    ]
    if not candidates:
        raise FileNotFoundError(f"AniPortrait did not create output/{pattern}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def extract_inputs(row: Dict[str, str], index: int, args, work: Path,
                   env: Dict[str, str]) -> Tuple[Path, Path]:
    input_video = Path(row["input_video_path"])
    if not input_video.exists():
        raise FileNotFoundError(input_video)
    token = f"{index:04d}_{uuid.uuid4().hex[:10]}"
    driving_audio = work / f"driving_{token}.wav"
    source_image = work / f"source_{token}.png"
    size = str(args.image_size)
    run([
        args.ffmpeg, "-y", "-i", str(input_video), "-vn", "-ac", "1",
        "-ar", "16000", "-c:a", "pcm_s16le", str(driving_audio),
    ], dry_run=args.dry_run, env=env)
    run([
        args.ffmpeg, "-y", "-ss", args.source_time, "-i", str(input_video),
        "-vf", f"crop='min(iw,ih)':'min(iw,ih)',scale={size}:{size}",
        "-frames:v", "1", str(source_image),
    ], dry_run=args.dry_run, env=env)
    return source_image, driving_audio


def postprocess(row: Dict[str, str], source_image: Path, driving_audio: Path, args,
                env: Dict[str, str]) -> None:
    output_video = Path(row["output_video_path"])
    output_video.parent.mkdir(parents=True, exist_ok=True)
    generated = find_generated_video(args.aniportrait_root, source_image.stem, driving_audio.stem)
    size = str(args.image_size)
    run([
        args.ffmpeg, "-y", "-i", str(generated), "-i", str(driving_audio),
        "-filter:v", f"crop=iw/3:ih:2*iw/3:0,scale={size}:{size}",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
        "-preset", "medium", "-crf", "18", "-c:a", "aac", "-shortest",
        str(output_video),
    ], dry_run=args.dry_run, env=env)
    if not args.dry_run and not output_video.exists():
        raise FileNotFoundError(f"AniPortrait did not create {output_video}")
    if not args.keep_intermediates and not args.dry_run:
        generated.unlink(missing_ok=True)


def selected_rows(args) -> List[Dict[str, str]]:
    rows = read_csv(args.tasks)
    missing = REQUIRED_COLUMNS - set(rows[0]) if rows else REQUIRED_COLUMNS
    if missing:
        raise ValueError(f"task CSV missing columns: {sorted(missing)}")
    rows = [row for row in rows if row["generator"] == args.generator]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < --num-shards")
    rows = rows[args.shard_index::args.num_shards]
    if args.resume:
        rows = [row for row in rows if not Path(row["output_video_path"]).exists()]
    if args.limit:
        rows = rows[:args.limit]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--generator", default="AniPortrait")
    parser.add_argument("--aniportrait-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--source-time", default="0")
    parser.add_argument("--accelerate", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--failures-csv", type=Path, required=True)
    args = parser.parse_args()

    args.tasks = args.tasks.resolve()
    args.aniportrait_root = args.aniportrait_root.resolve()
    if not args.dry_run:
        validate_install(args.aniportrait_root, args.accelerate)
    env = runtime_env(args.device)
    rows = selected_rows(args)
    if not rows:
        write_csv(args.failures_csv, [])
        print("[aniportrait-batch] done ok=0 skipped=all failed=0")
        return

    failures: List[Dict[str, str]] = []
    prepared: List[Tuple[Dict[str, str], Path, Path]] = []
    with tempfile.TemporaryDirectory(prefix=f"rq1_aniportrait_batch_s{args.shard_index}_") as raw_tmp:
        tmp = Path(raw_tmp)
        for index, row in enumerate(rows):
            try:
                source_image, driving_audio = extract_inputs(row, index, args, tmp, env)
                prepared.append((row, source_image, driving_audio))
            except Exception as exc:
                fail = dict(row)
                fail["error"] = str(exc)[-1000:]
                failures.append(fail)
                print(f"[aniportrait-batch] prepare FAILED {row['task_id']}: {exc}", flush=True)

        if prepared:
            config = tmp / f"animation_audio_batch_s{args.shard_index}.yaml"
            write_audio_config(config, args.aniportrait_root, [(s, a) for _, s, a in prepared])
            command = [
                sys.executable, "-m", "scripts.audio2vid", "--config", str(config),
                "-W", str(args.image_size), "-H", str(args.image_size),
                "--steps", str(args.steps),
            ]
            if args.accelerate:
                command.append("-acc")
            try:
                run(command, cwd=args.aniportrait_root, dry_run=args.dry_run, env=env)
            except Exception as exc:
                for row, _source, _audio in prepared:
                    fail = dict(row)
                    fail["error"] = f"batch audio2vid failed: {str(exc)[-900:]}"
                    failures.append(fail)
                    print(f"[aniportrait-batch] audio2vid FAILED {row['task_id']}: {exc}", flush=True)
                prepared = []

        n_ok = 0
        if args.dry_run:
            n_ok = len(prepared)
            for row, _source_image, _driving_audio in prepared:
                print(f"[aniportrait-batch] dry-ok {row['task_id']}", flush=True)
        else:
            for row, source_image, driving_audio in prepared:
                try:
                    postprocess(row, source_image, driving_audio, args, env)
                    n_ok += 1
                    print(f"[aniportrait-batch] ok     {row['task_id']}", flush=True)
                except Exception as exc:
                    fail = dict(row)
                    fail["error"] = str(exc)[-1000:]
                    failures.append(fail)
                    print(f"[aniportrait-batch] post FAILED {row['task_id']}: {exc}", flush=True)

    write_csv(args.failures_csv, failures)
    skipped = "resume-existing" if args.resume else "0"
    print(f"[aniportrait-batch] done ok={n_ok} skipped={skipped} failed={len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
