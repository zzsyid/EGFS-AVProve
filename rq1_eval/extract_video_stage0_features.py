#!/usr/bin/env python3
"""Extract ArcFace identity and CLIP visual-event features for M2 Stage-0."""

from __future__ import annotations

import argparse
import copy
import csv
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from extract_features import ArcFaceExtractor, CLIPVisualExtractor, l2norm


CHANNELS = ["id_video", "evt_video"]
META_COLUMNS = [
    "attack_level", "attack_family", "attack_scope",
    "target_start_sec", "target_end_sec", "generator_role",
    "attack_variant", "splice_duration_sec", "splice_position",
    "codec_bandwidth", "enhancement_model", "attack_language",
    "video_crf", "audio_bitrate",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class VideoExtractor:
    def __init__(self, args) -> None:
        self.id_video = ArcFaceExtractor(args.device, n_frames=args.face_frames)
        self.evt_video = CLIPVisualExtractor(
            args.device, args.clip_model, n_frames=args.clip_frames,
        )

def extract_rows(args, rows: List[Dict[str, str]], worker: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    extractor = VideoExtractor(args)
    out_root = Path(args.out_root)
    completed, failures = [], []
    for index, row in enumerate(rows, start=1):
        try:
            sample_id = row["sample_id"]
            video_path = Path(row["video_path"])
            sample_root = out_root / sample_id
            paths = {channel: sample_root / f"{channel}.npy" for channel in CHANNELS}
            sample_root.mkdir(parents=True, exist_ok=True)
            channel_errors = {}
            for channel in CHANNELS:
                if args.resume and paths[channel].exists():
                    continue
                try:
                    vector = getattr(extractor, channel)(video_path)
                    np.save(paths[channel], l2norm(vector))
                except Exception as exc:
                    channel_errors[channel] = str(exc)[-1000:]
            available = {channel: paths[channel].exists() for channel in CHANNELS}
            if not any(available.values()):
                details = "; ".join(
                    f"{channel}: {error}" for channel, error in channel_errors.items()
                )
                raise RuntimeError(details or "no video feature channel available")
            rec = {
                "sample_id": sample_id,
                "enroll_id": row["enroll_id"],
                "source_id": row["source_id"],
                "population": row["population"],
                "cond": row["cond"],
                "generator": row["generator"],
                "split": row["split"],
            }
            for col in META_COLUMNS:
                if col in row:
                    rec[col] = row[col]
            for channel in CHANNELS:
                rec[f"{channel}_path"] = str(paths[channel].resolve()) if available[channel] else ""
                rec[f"{channel}_available"] = str(available[channel])
                rec[f"{channel}_error"] = channel_errors.get(channel, "")
            completed.append(rec)
            status = "partial" if channel_errors else "ok     "
            detail = f" errors={channel_errors}" if channel_errors else ""
            print(f"[{worker} {index}/{len(rows)}] {status} {sample_id}{detail}", flush=True)
        except Exception as exc:
            rec = dict(row)
            rec["error"] = str(exc)[-1000:]
            failures.append(rec)
            print(f"[{worker} {index}/{len(rows)}] FAILED {row.get('sample_id', '?')}: {exc}", flush=True)
    return completed, failures


def worker_main(args, rows: List[Dict[str, str]], device: str, worker_id: int,
                success_csv: str, failure_csv: str) -> None:
    args = copy.deepcopy(args)
    args.device = device
    name = f"gpu{worker_id}:{device}"
    print(f"[{name}] starting rows={len(rows)}", flush=True)
    completed, failures = extract_rows(args, rows, name)
    write_csv(Path(success_csv), completed)
    write_csv(Path(failure_csv), failures)
    print(f"[{name}] done ok={len(completed)} failed={len(failures)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--face-frames", type=int, default=16)
    parser.add_argument("--clip-frames", type=int, default=8)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    args = parser.parse_args()

    rows = read_csv(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    devices = [device.strip() for device in args.devices.split(",") if device.strip()]
    if not devices:
        raise ValueError("--devices must list at least one device")
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    shards = [rows[index::len(devices)] for index in range(len(devices))]

    if len(devices) == 1:
        args.device = devices[0]
        completed, failures = extract_rows(args, rows, f"gpu0:{args.device}")
    else:
        ctx = mp.get_context("spawn")
        jobs = []
        shard_paths = []
        for worker_id, (device, shard) in enumerate(zip(devices, shards)):
            success = args.out_manifest.with_name(f"{args.out_manifest.stem}.worker{worker_id}.csv")
            failure = args.out_manifest.with_name(f"{args.out_manifest.stem}.worker{worker_id}.failures.csv")
            for stale in (success, failure):
                if stale.exists():
                    stale.unlink()
            shard_paths.append((success, failure))
            job = ctx.Process(
                target=worker_main,
                args=(args, shard, device, worker_id, str(success), str(failure)),
            )
            job.start()
            jobs.append(job)
        for job in jobs:
            job.join()
        failed_jobs = [job.pid for job in jobs if job.exitcode != 0]
        if failed_jobs:
            raise RuntimeError(f"video feature workers failed: pids={failed_jobs}")
        completed, failures = [], []
        for success, failure in shard_paths:
            if success.exists():
                completed.extend(read_csv(success))
            if failure.exists():
                failures.extend(read_csv(failure))

    write_csv(args.out_manifest, completed)
    failure_csv = args.out_manifest.with_suffix(".failures.csv")
    if failures:
        write_csv(failure_csv, failures)
        print(f"[m2-extract] failures -> {failure_csv}")
    elif failure_csv.exists():
        failure_csv.unlink()
    print(f"[m2-extract] rows={len(completed)} failed={len(failures)} -> {args.out_manifest}")


if __name__ == "__main__":
    main()
