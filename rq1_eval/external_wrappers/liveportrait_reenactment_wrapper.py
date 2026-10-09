#!/usr/bin/env python3
"""Reenact a source portrait with a donor video through LivePortrait."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional


def run(
    command: List[str],
    cwd: Optional[Path] = None,
    dry_run: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> None:
    print("[liveportrait-wrapper]", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=str(cwd) if cwd else None, check=True, env=env)


def bind_cuda_visible_devices(env: Dict[str, str], device: str) -> None:
    if device.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        print(f"[liveportrait-wrapper] bind CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} for {device}", flush=True)


def liveportrait_runtime_env(device: str = "cuda:0") -> Dict[str, str]:
    env = os.environ.copy()
    bind_cuda_visible_devices(env, device)
    # The server-side ONNXRuntime build expects CUDA 11 sonames. The conda
    # environment also installs activation hooks, but the benchmark calls its
    # Python executable directly, so inject the same runtime paths here.
    candidates = [
        "/path/to/runtime/anaconda3/pkgs/libcublas-11.11.3.6-0/lib",
        "/path/to/runtime/anaconda3/pkgs/libcufft-10.9.0.58-0/lib",
        "/path/to/runtime/anaconda3/pkgs/libcurand-10.3.4.101-0/lib",
        "/path/to/runtime/anaconda3/envs/cxr/lib/python3.11/site-packages/nvidia/cuda_runtime/lib",
        str(Path(sys.prefix) / "lib/python3.10/site-packages/nvidia/cudnn/lib"),
    ]
    existing = [path for path in candidates if Path(path).exists()]
    if existing:
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(existing + ([current] if current else []))
    return env


def validate_install(root: Path) -> None:
    required = [
        "inference.py",
        "pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth",
        "pretrained_weights/liveportrait/base_models/motion_extractor.pth",
        "pretrained_weights/liveportrait/base_models/spade_generator.pth",
        "pretrained_weights/liveportrait/base_models/warping_module.pth",
        "pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth",
    ]
    missing = [relative for relative in required if not (root / relative).exists()]
    if missing:
        details = "\n  - ".join(missing)
        raise FileNotFoundError(f"incomplete LivePortrait install:\n  - {details}")


def find_generated_video(output_dir: Path) -> Path:
    candidates = [
        path for path in output_dir.rglob("*.mp4")
        if not path.stem.endswith("_concat")
    ]
    if not candidates:
        raise FileNotFoundError(f"LivePortrait did not create a generated mp4 in {output_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--donor-video", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--liveportrait-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-times", default="0,1,2,3",
                        help="comma-separated source-frame timestamps retried on failure")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        for path in (args.input_video, args.donor_video):
            if not path.exists():
                raise FileNotFoundError(path)
        validate_install(args.liveportrait_root)
        args.output_video.parent.mkdir(parents=True, exist_ok=True)

    source_times = [text.strip() for text in args.source_times.split(",") if text.strip()]
    if not source_times:
        raise ValueError("--source-times must contain at least one timestamp")

    token = uuid.uuid4().hex[:12]
    with tempfile.TemporaryDirectory(prefix="rq1_liveportrait_") as raw_tmp:
        tmp = Path(raw_tmp)
        generated = None
        for attempt, source_time in enumerate(source_times, start=1):
            source_image = tmp / f"source_{token}_{attempt}.png"
            attempt_dir = tmp / f"animations_{attempt}"
            run([
                args.ffmpeg, "-y", "-ss", source_time, "-i", str(args.input_video),
                "-vf", "crop='min(iw,ih)':'min(iw,ih)',scale=512:512",
                "-frames:v", "1", str(source_image),
            ], dry_run=args.dry_run)
            command = [
                sys.executable, "inference.py",
                "-s", str(source_image),
                "-d", str(args.donor_video),
                "-o", str(attempt_dir),
                "--flag_crop_driving_video",
            ]
            try:
                run(
                    command, cwd=args.liveportrait_root, dry_run=args.dry_run,
                    env=liveportrait_runtime_env(args.device),
                )
                if args.dry_run:
                    return
                generated = find_generated_video(attempt_dir)
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                if attempt == len(source_times):
                    raise
                print(
                    f"[liveportrait-wrapper] attempt {attempt} failed; "
                    f"retry source frame at {source_times[attempt]} sec",
                    flush=True,
                )
        if generated is None:
            raise FileNotFoundError("LivePortrait did not create a generated video")

        # Remux donor audio explicitly. This makes the AV replacement contract
        # stable across LivePortrait versions and output settings.
        run([
            args.ffmpeg, "-y", "-i", str(generated), "-i", str(args.donor_video),
            "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac",
            "-shortest", str(args.output_video),
        ])

    if not args.output_video.exists():
        raise FileNotFoundError(f"LivePortrait did not create {args.output_video}")


if __name__ == "__main__":
    main()
