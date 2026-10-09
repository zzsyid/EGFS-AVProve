#!/usr/bin/env python3
"""Redraw a registered AV clip through an installed EchoMimicV3-Flash model."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional


def run(command: List[str], cwd: Optional[Path] = None, dry_run: bool = False) -> None:
    print("[echomimicv3-wrapper]", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def bind_cuda_visible_devices(device: str) -> None:
    if not device or device == "cpu":
        return
    if device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device.rsplit(":", 1)[1]
        print(
            f"[echomimicv3-wrapper] binding child processes to CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}",
            flush=True,
        )
    elif device == "cuda":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    else:
        raise ValueError(f"unsupported EchoMimicV3 device: {device}")


def validate_install(root: Path) -> None:
    required = [
        "infer_flash.py",
        "flash/Wan2.1-Fun-V1.1-1.3B-InP",
        "flash/chinese-wav2vec2-base",
        "flash/transformer/diffusion_pytorch_model.safetensors",
    ]
    missing = [relative for relative in required if not (root / relative).exists()]
    if missing:
        details = "\n  - ".join(missing)
        raise FileNotFoundError(f"incomplete EchoMimicV3-Flash install:\n  - {details}")


def find_generated_video(output_dir: Path, source_stem: str) -> Path:
    preferred = output_dir / f"{source_stem}_output.mp4"
    if preferred.exists():
        return preferred
    candidates = list(output_dir.rglob("*.mp4"))
    if not candidates:
        raise FileNotFoundError(f"EchoMimicV3 did not create an mp4 in {output_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--echomimicv3-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video-length", type=int, default=138,
                        help="bounded pilot frame count; use long-video CFG separately for longer clips")
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--prompt",
        default="A person is speaking naturally to the camera with realistic facial motion.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bind_cuda_visible_devices(args.device)

    if args.video_length < 1:
        raise ValueError("--video-length must be positive")
    if not args.dry_run:
        if not args.input_video.exists():
            raise FileNotFoundError(args.input_video)
        validate_install(args.echomimicv3_root)
        args.output_video.parent.mkdir(parents=True, exist_ok=True)

    model_root = args.echomimicv3_root / "flash" / "Wan2.1-Fun-V1.1-1.3B-InP"
    wav2vec_root = args.echomimicv3_root / "flash" / "chinese-wav2vec2-base"
    transformer_path = args.echomimicv3_root / "flash" / "transformer" / "diffusion_pytorch_model.safetensors"
    config_path = args.echomimicv3_root / "config" / "config.yaml"

    token = uuid.uuid4().hex[:12]
    with tempfile.TemporaryDirectory(prefix="rq1_echomimicv3_") as raw_tmp:
        tmp = Path(raw_tmp)
        source_image = tmp / f"source_{token}.png"
        driving_audio = tmp / f"driving_{token}.wav"
        save_dir = tmp / "output"
        run([
            args.ffmpeg, "-y", "-ss", "0", "-i", str(args.input_video),
            "-vf", "crop='min(iw,ih)':'min(iw,ih)',scale=768:768",
            "-frames:v", "1", str(source_image),
        ], dry_run=args.dry_run)
        run([
            args.ffmpeg, "-y", "-i", str(args.input_video), "-vn", "-ac", "1",
            "-ar", "16000", "-c:a", "pcm_s16le", str(driving_audio),
        ], dry_run=args.dry_run)
        launcher = Path(__file__).with_name("echomimicv3_infer_launcher.py")
        run([
            sys.executable, str(launcher),
            "--echomimicv3-root", str(args.echomimicv3_root),
            "--",
            "--image_path", str(source_image),
            "--audio_path", str(driving_audio),
            "--save_path", str(save_dir),
            "--config_path", str(config_path),
            "--model_name", str(model_root),
            "--wav2vec_model_dir", str(wav2vec_root),
            "--transformer_path", str(transformer_path),
            "--prompt", args.prompt,
            "--video_length", str(args.video_length),
            "--num_inference_steps", str(args.sample_steps),
            "--seed", str(args.seed),
        ], dry_run=args.dry_run)
        if args.dry_run:
            return
        generated = find_generated_video(save_dir, source_image.stem)
        shutil.copy2(generated, args.output_video)

    if not args.output_video.exists():
        raise FileNotFoundError(f"EchoMimicV3 did not create {args.output_video}")


if __name__ == "__main__":
    main()
