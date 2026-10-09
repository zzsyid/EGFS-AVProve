#!/usr/bin/env python3
"""Regenerate a talking-head video with an installed AniPortrait repository."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional


def run(command: List[str], cwd: Optional[Path] = None, dry_run: bool = False,
        env: Optional[dict] = None) -> None:
    print("[aniportrait-wrapper]", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(
            command, cwd=str(cwd) if cwd else None, check=True, env=env,
        )



def device_visible_id(device: str) -> Optional[str]:
    if not device or device == "cpu":
        return None
    if device.startswith("cuda:"):
        return device.rsplit(":", 1)[1]
    if device == "cuda":
        return "0"
    raise ValueError(f"unsupported AniPortrait device: {device}")

def yaml_string(path: Path) -> str:
    # JSON strings are valid YAML scalars and safely preserve spaces in paths.
    return json.dumps(str(path))


def write_audio_config(path: Path, root: Path, source_image: Path,
                       driving_audio: Path) -> None:
    text = f"""\
pretrained_base_model_path: {yaml_string(root / "pretrained_model/stable-diffusion-v1-5")}
pretrained_vae_path: {yaml_string(root / "pretrained_model/sd-vae-ft-mse")}
image_encoder_path: {yaml_string(root / "pretrained_model/image_encoder")}
denoising_unet_path: {yaml_string(root / "pretrained_model/denoising_unet.pth")}
reference_unet_path: {yaml_string(root / "pretrained_model/reference_unet.pth")}
pose_guider_path: {yaml_string(root / "pretrained_model/pose_guider.pth")}
motion_module_path: {yaml_string(root / "pretrained_model/motion_module.pth")}
audio_inference_config: {yaml_string(root / "configs/inference/inference_audio.yaml")}
inference_config: {yaml_string(root / "configs/inference/inference_v2.yaml")}
weight_dtype: "fp16"
test_cases:
  {yaml_string(source_image)}:
    - {yaml_string(driving_audio)}
"""
    path.write_text(text, encoding="utf-8")


def find_generated_video(root: Path, source_stem: str, audio_stem: str) -> Path:
    pattern = f"{source_stem}_{audio_stem}_*.mp4"
    candidates = [
        path for path in (root / "output").rglob(pattern)
        if not path.name.endswith("_noaudio.mp4")
    ]
    if not candidates:
        raise FileNotFoundError(f"AniPortrait did not create output/{pattern}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


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
        details = "\n  - ".join(missing)
        raise FileNotFoundError(f"incomplete AniPortrait install:\n  - {details}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--aniportrait-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--source-times", default="0,1,2,3",
                        help="comma-separated frame timestamps retried when face detection fails")
    parser.add_argument("--accelerate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.input_video.exists() and not args.dry_run:
        raise FileNotFoundError(args.input_video)
    if not args.dry_run:
        validate_install(args.aniportrait_root, args.accelerate)
        args.output_video.parent.mkdir(parents=True, exist_ok=True)

    runtime_env = os.environ.copy()
    visible_id = device_visible_id(args.device)
    if visible_id is not None:
        runtime_env["CUDA_VISIBLE_DEVICES"] = visible_id
        print(
            f"[aniportrait-wrapper] binding child processes to CUDA_VISIBLE_DEVICES={visible_id}",
            flush=True,
        )

    token = uuid.uuid4().hex[:12]
    with tempfile.TemporaryDirectory(prefix="rq1_aniportrait_") as raw_tmp:
        tmp = Path(raw_tmp)
        driving_audio = tmp / f"driving_{token}.wav"
        size = str(args.image_size)
        run([
            args.ffmpeg, "-y", "-i", str(args.input_video), "-vn", "-ac", "1",
            "-ar", "16000", "-c:a", "pcm_s16le", str(driving_audio),
        ], dry_run=args.dry_run, env=runtime_env)

        source_times = [text.strip() for text in args.source_times.split(",") if text.strip()]
        if not source_times:
            raise ValueError("--source-times must contain at least one timestamp")
        generated = None
        for attempt, source_time in enumerate(source_times, start=1):
            source_image = tmp / f"source_{token}_{attempt}.png"
            config = tmp / f"animation_audio_{token}_{attempt}.yaml"
            run([
                args.ffmpeg, "-y", "-ss", source_time, "-i", str(args.input_video),
                "-vf", f"crop='min(iw,ih)':'min(iw,ih)',scale={size}:{size}",
                "-frames:v", "1", str(source_image),
            ], dry_run=args.dry_run, env=runtime_env)
            write_audio_config(config, args.aniportrait_root, source_image, driving_audio)

            command = [
                sys.executable, "-m", "scripts.audio2vid", "--config", str(config),
                "-W", size, "-H", size, "--steps", str(args.steps),
            ]
            if args.accelerate:
                command.append("-acc")
            try:
                run(command, cwd=args.aniportrait_root, dry_run=args.dry_run, env=runtime_env)
                if args.dry_run:
                    return
                generated = find_generated_video(
                    args.aniportrait_root, source_image.stem, driving_audio.stem,
                )
                break
            except subprocess.CalledProcessError:
                if attempt == len(source_times):
                    raise
                print(
                    f"[aniportrait-wrapper] attempt {attempt} failed; "
                    f"retry source frame at {source_times[attempt]} sec",
                    flush=True,
                )
        if generated is None:
            raise FileNotFoundError("AniPortrait did not create a generated video")

        # AniPortrait's CLI saves [reference | landmark | generated] as one
        # horizontal grid. Keep only the generated right-most panel.
        run([
            args.ffmpeg, "-y", "-i", str(generated), "-i", str(driving_audio),
            "-filter:v", f"crop=iw/3:ih:2*iw/3:0,scale={size}:{size}",
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
            "-preset", "medium", "-crf", "18", "-c:a", "aac", "-shortest",
            str(args.output_video),
        ], env=runtime_env)
        generated.unlink()

    if not args.output_video.exists():
        raise FileNotFoundError(f"AniPortrait did not create {args.output_video}")


if __name__ == "__main__":
    main()
