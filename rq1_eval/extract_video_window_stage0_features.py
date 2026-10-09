#!/usr/bin/env python3
"""Extract ArcFace and CLIP features on fixed video windows for M2 localization."""

from __future__ import annotations

import argparse
import copy
import csv
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from extract_features import ArcFaceExtractor, CLIPVisualExtractor


KEEP_CONDS = {
    "clean",
    "L1",
    "video_splice",
    "face_swap",
    "fake_video",
    "fake_av",
    "modal_replacement",
    "av_modal_replacement",
}
META_COLUMNS = [
    "sample_id", "enroll_id", "source_id", "population", "cond", "generator",
    "split", "video_path", "attack_level", "attack_family", "attack_scope",
    "target_start_sec", "target_end_sec", "generator_role",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def duration_sec(video_path: Path) -> float:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if frames <= 0 or fps <= 0:
        raise RuntimeError(f"cannot determine video duration: {video_path}")
    return frames / fps


def window_starts(duration: float, span: float, hop: float) -> np.ndarray:
    if duration < span:
        return np.asarray([0.0], dtype=np.float32)
    count = int(np.floor((duration - span) / hop)) + 1
    return np.asarray([index * hop for index in range(count)], dtype=np.float32)


class WindowExtractor:
    def __init__(self, args) -> None:
        self.id_video = ArcFaceExtractor(args.device, n_frames=args.face_frames)
        self.evt_video = CLIPVisualExtractor(
            args.device, args.clip_model, n_frames=args.clip_frames,
        )
        self.span = args.window_span_sec
        self.hop = args.window_hop_sec

    def __call__(self, video_path: Path) -> Dict[str, np.ndarray]:
        starts = window_starts(duration_sec(video_path), self.span, self.hop)
        id_vectors, evt_vectors, id_available, evt_available = [], [], [], []
        for start in starts:
            end = float(start + self.span)
            try:
                id_vectors.append(self.id_video.extract_window(video_path, float(start), end))
                id_available.append(True)
            except Exception:
                id_vectors.append(np.zeros(512, dtype=np.float32))
                id_available.append(False)
            try:
                evt_vectors.append(self.evt_video.extract_window(video_path, float(start), end))
                evt_available.append(True)
            except Exception:
                evt_vectors.append(np.zeros(512, dtype=np.float32))
                evt_available.append(False)
        return {
            "starts_sec": starts,
            "id_video": np.stack(id_vectors).astype(np.float32),
            "evt_video": np.stack(evt_vectors).astype(np.float32),
            "id_video_available": np.asarray(id_available, dtype=np.bool_),
            "evt_video_available": np.asarray(evt_available, dtype=np.bool_),
        }


def extract_rows(args, rows: List[Dict[str, str]], worker: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    extractor = WindowExtractor(args)
    completed, failures = [], []
    out_root = Path(args.out_root)
    for index, row in enumerate(rows, start=1):
        try:
            sample_id = row["sample_id"]
            output = out_root / sample_id / "video_windows.npz"
            if not (args.resume and output.exists()):
                features = extractor(Path(row["video_path"]))
                output.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output, **features)
            data = np.load(output)
            rec = {column: row.get(column, "") for column in META_COLUMNS}
            rec["window_feature_path"] = str(output.resolve())
            rec["window_count"] = str(len(data["starts_sec"]))
            rec["id_window_available"] = str(int(data["id_video_available"].sum()))
            rec["evt_window_available"] = str(int(data["evt_video_available"].sum()))
            completed.append(rec)
            print(f"[{worker} {index}/{len(rows)}] ok      {sample_id}", flush=True)
        except Exception as exc:
            rec = dict(row)
            rec["error"] = str(exc)[-1000:]
            failures.append(rec)
            print(f"[{worker} {index}/{len(rows)}] FAILED  {row.get('sample_id', '?')}: {exc}", flush=True)
    return completed, failures


def worker_main(args, rows: List[Dict[str, str]], device: str, worker_id: int,
                success_csv: str, failure_csv: str) -> None:
    args = copy.deepcopy(args)
    args.device = device
    completed, failures = extract_rows(args, rows, f"gpu{worker_id}:{device}")
    write_csv(Path(success_csv), completed)
    write_csv(Path(failure_csv), failures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--window-span-sec", type=float, default=1.0)
    parser.add_argument("--window-hop-sec", type=float, default=0.5)
    parser.add_argument("--face-frames", type=int, default=8)
    parser.add_argument("--clip-frames", type=int, default=4)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="extract every manifest row instead of the historical M2 cond whitelist",
    )
    args = parser.parse_args()

    if args.window_span_sec <= 0 or args.window_hop_sec <= 0:
        raise ValueError("window span and hop must be positive")
    raw_rows = read_csv(args.manifest)
    rows = raw_rows if args.keep_all else [row for row in raw_rows if row.get("cond") in KEEP_CONDS]
    devices = [device.strip() for device in args.devices.split(",") if device.strip()]
    if not devices:
        raise ValueError("--devices must list at least one device")
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    shards = [rows[index::len(devices)] for index in range(len(devices))]

    if len(devices) == 1:
        args.device = devices[0]
        completed, failures = extract_rows(args, rows, f"gpu0:{devices[0]}")
    else:
        ctx = mp.get_context("spawn")
        jobs, shard_paths = [], []
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
        if any(job.exitcode != 0 for job in jobs):
            raise RuntimeError("one or more video-window feature workers failed")
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
    elif failure_csv.exists():
        failure_csv.unlink()
    print(f"[m2-window-extract] rows={len(completed)} failed={len(failures)} -> {args.out_manifest}")


if __name__ == "__main__":
    main()
