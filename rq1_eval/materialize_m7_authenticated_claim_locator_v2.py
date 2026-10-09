#!/usr/bin/env python3
"""Materialize Audio32/Video24 media from authenticated M7 v2 locators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from authenticated_locator_carrier import (
    embed_videoseal_full_duration,
    embed_packet_beacon,
    load_signed_manifests,
    mux_video_audio,
    safe_name,
    write_csv,
)


POLICY = "m7-authenticated-claim-locator-v2"


def locator_hex(manifest: Dict[str, Any]) -> str:
    locator = str(manifest.get("authenticated_claim_locator_hex", "")).lower()
    if not locator:
        audio = dict(manifest.get("payloads", {}).get("audio", {}))
        bound = dict(audio.get("bound_fragments", {}))
        locator = str(bound.get("authenticated_claim_locator", "")).lower()
    if len(locator) != 8 or any(char not in "0123456789abcdef" for char in locator):
        raise ValueError("manifest lacks a valid 32-bit authenticated claim locator")
    return locator


def locator_bits(locator: str) -> List[int]:
    return [int(bit) for bit in bin(int(locator, 16))[2:].zfill(32)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signed-payload-registry", type=Path, required=True)
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=Path("/path/to/runtime/anaconda3/envs/iap/bin/ffmpeg"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audio-bitrate", default="128k")
    parser.add_argument("--audio-master-key", default="tdsc-m7-audio-packet-beacon-v1")
    parser.add_argument("--audio-embed-window-sec", type=float, default=0.75)
    parser.add_argument("--audio-embed-hop-sec", type=float, default=0.75)
    parser.add_argument("--audio-edge-guard-sec", type=float, default=0.2)
    parser.add_argument("--audio-watermark-scale", type=float, default=1.0)
    parser.add_argument("--video-pointer-bits", type=int, default=24)
    parser.add_argument("--video-resize", type=int, default=256)
    parser.add_argument("--video-batch-frames", type=int, default=8)
    parser.add_argument("--video-crf", type=int, default=23)
    parser.add_argument("--videoseal-scaling-w", type=float, default=0.8)
    parser.add_argument("--frame-cache-root", type=Path, default=None)
    parser.add_argument(
        "--frame-cache-mode",
        choices=["off", "readwrite", "readonly"],
        default="off",
    )
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.video_pointer_bits != 24:
        raise ValueError("v2 currently requires a 24-bit video locator")

    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    manifests = [
        row
        for index, row in enumerate(
            load_signed_manifests(args.signed_payload_registry, args.public_key)
        )
        if index % args.num_shards == args.shard_index
    ]
    if args.limit:
        manifests = manifests[: args.limit]

    from audioseal import AudioSeal
    import videoseal

    audio_generator = AudioSeal.load_generator("audioseal_wm_16bits").to(device).eval()
    audio_detector = AudioSeal.load_detector("audioseal_detector_16bits").to(device).eval()
    video_model = videoseal.load("videoseal")
    video_model.eval().to(device)

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    args.out_root.mkdir(parents=True, exist_ok=True)
    for index, manifest in enumerate(manifests, start=1):
        sample_id = str(manifest.get("sample_id", ""))
        enroll_id = str(manifest.get("enroll_id", sample_id))
        record_hash = str(manifest.get("registry_record_sha256", ""))
        input_video = Path(str(manifest.get("video_path", "")))
        output = args.out_root / "media" / f"{safe_name(sample_id)}.mp4"
        work_dir = args.out_root / "work" / safe_name(sample_id)
        video_only = work_dir / "videoseal_repeat24_locator_v2_video_only.mp4"
        try:
            locator = locator_hex(manifest)
            if args.resume and output.exists():
                status = "skipped"
                meta: Dict[str, Any] = {}
            else:
                if not input_video.exists():
                    raise FileNotFoundError(str(input_video))
                audio = embed_packet_beacon(
                    audio_generator,
                    audio_detector,
                    input_video,
                    locator_bits(locator),
                    args.audio_embed_window_sec,
                    args.audio_embed_hop_sec,
                    args.audio_watermark_scale,
                    args.audio_edge_guard_sec,
                    device,
                    args.ffmpeg,
                    work_dir / "audio32_authenticated_locator_v2",
                    args.audio_master_key,
                )
                video = embed_videoseal_full_duration(
                    video_model,
                    input_video,
                    locator,
                    24,
                    device,
                    args.video_resize,
                    args.video_batch_frames,
                    args.videoseal_scaling_w,
                    args.ffmpeg,
                    video_only,
                    args.video_crf,
                    args.frame_cache_root,
                    args.frame_cache_mode,
                )
                mux_video_audio(
                    video_only,
                    Path(str(audio["watermarked_wav"])),
                    output,
                    args.ffmpeg,
                    args.audio_bitrate,
                )
                status = "ok"
                meta = {
                    "audio32_tag_hex": locator,
                    "audio_duration_sec": float(audio["duration_sec"]),
                    "audio_embed_windows": int(audio["embed_windows"]),
                    **video,
                }
            rows.append(
                {
                    "sample_id": sample_id,
                    "enroll_id": enroll_id,
                    "source_id": str(manifest.get("source_id", "")),
                    "original_video_path": str(input_video),
                    "embedded_video_path": str(output),
                    "m7_payload_registry_record_sha256": record_hash,
                    "authenticated_claim_locator_hex": locator,
                    "claim_version": manifest.get("claim_version", ""),
                    "claim_family_sha256": manifest.get("claim_family_sha256", ""),
                    "m4_registry_record_sha256": str(
                        manifest.get("m4_registry_record_sha256", "")
                    ),
                    "semantic_registry_status": str(
                        manifest.get("semantic_registry_status", "")
                    ),
                    "signed_payload_registry": str(args.signed_payload_registry),
                    "public_key": str(args.public_key),
                    "m7_carrier_policy": POLICY,
                    "audio_carrier": "audio32_authenticated_claim_locator_v2",
                    "video_carrier": "videoseal_repeat24_authenticated_claim_locator_v2",
                    "status": status,
                    **meta,
                }
            )
            print(
                f"[m7-locator-v2 {index}/{len(manifests)}] {status:7s} {sample_id}",
                flush=True,
            )
        except Exception as exc:
            failures.append(
                {
                    "sample_id": sample_id,
                    "enroll_id": enroll_id,
                    "source_id": str(manifest.get("source_id", "")),
                    "original_video_path": str(input_video),
                    "m7_payload_registry_record_sha256": record_hash,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"[m7-locator-v2 {index}/{len(manifests)}] FAILED {sample_id}: {exc}",
                flush=True,
            )

    write_csv(args.out_manifest, rows)
    failures_csv = args.out_manifest.with_suffix(".failures.csv")
    write_csv(failures_csv, failures)
    summary = {
        "policy": POLICY,
        "signed_payload_registry": str(args.signed_payload_registry),
        "public_key": str(args.public_key),
        "requested": len(manifests),
        "ok": len(rows),
        "failed": len(failures),
        "out_manifest": str(args.out_manifest),
        "failures_csv": str(failures_csv),
        "audio": "audio32_authenticated_claim_locator_v2",
        "video": "videoseal_repeat24_authenticated_claim_locator_v2",
    }
    args.out_manifest.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
