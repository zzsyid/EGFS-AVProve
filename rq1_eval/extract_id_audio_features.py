#!/usr/bin/env python3
"""Extract only ECAPA id_audio features for M1 speaker-identity evaluation.

This avoids coupling the audio-identity VC experiment to ArcFace/CLIP/Whisper
failures from the four-channel extractor.
"""

from __future__ import annotations

import argparse
import copy
import csv
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from extract_features import ECAPAExtractor, META_COLUMNS, l2norm, resolve_path


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def require_cols(rows: List[Dict[str, str]], cols: List[str]) -> None:
    if not rows:
        raise ValueError("input manifest is empty")
    missing = [col for col in cols if col not in rows[0]]
    if missing:
        raise ValueError(f"input manifest missing columns: {missing}")


def extract_rows(args, in_rows: List[Dict[str, str]], worker_name: str = "worker"):
    path_root = Path(args.path_root or ".").resolve()
    out_root = Path(args.out_root)
    extractor = ECAPAExtractor(args.device, args.ecapa_source, args.ecapa_savedir)
    feature_rows = []
    failures = []

    for index, raw in enumerate(in_rows, start=1):
        row = dict(raw)
        try:
            if not row.get("enroll_id"):
                row["enroll_id"] = row["sample_id"]
            sid = row["sample_id"]
            video_path = resolve_path(row["video_path"], path_root)
            sample_dir = out_root / sid
            path = sample_dir / "id_audio.npy"
            if not (args.resume and path.exists()):
                vec = extractor(video_path)
                sample_dir.mkdir(parents=True, exist_ok=True)
                np.save(path, l2norm(vec))
            out = {
                "sample_id": sid,
                "enroll_id": row["enroll_id"],
                "source_id": row["source_id"],
                "population": row["population"],
                "cond": row["cond"],
                "generator": row["generator"],
                "split": row["split"],
            }
            for col in META_COLUMNS:
                if col in row:
                    out[col] = row[col]
            out["id_audio_path"] = str(path.resolve())
            feature_rows.append(out)
            print(f"[{worker_name} {index}/{len(in_rows)}] ok     {sid}", flush=True)
        except Exception as exc:
            fail = dict(raw)
            fail["error"] = str(exc)[-500:]
            failures.append(fail)
            print(
                f"[{worker_name} {index}/{len(in_rows)}] FAILED "
                f"{raw.get('sample_id', '?')}: {fail['error']}",
                flush=True,
            )
    return feature_rows, failures


def worker_main(args, in_rows: List[Dict[str, str]], device: str,
                worker_id: int, success_csv: str, failure_csv: str) -> None:
    worker_args = copy.deepcopy(args)
    worker_args.device = device
    name = f"gpu{worker_id}:{device}"
    print(f"[{name}] starting shard with {len(in_rows)} rows", flush=True)
    feature_rows, failures = extract_rows(worker_args, in_rows, worker_name=name)
    write_csv(Path(success_csv), feature_rows)
    if failures:
        write_csv(Path(failure_csv), failures)
    print(f"[{name}] done ok={len(feature_rows)} failed={len(failures)}", flush=True)


def parse_devices(device: str, devices: str) -> List[str]:
    out = [d.strip() for d in devices.split(",") if d.strip()] if devices else [device]
    if not out:
        raise ValueError("no devices configured")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--path-root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ecapa-source", default="speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument("--ecapa-savedir", default="pretrained_models/ecapa_voxceleb")
    args = parser.parse_args()

    in_rows = read_csv(Path(args.manifest))
    require_cols(
        in_rows,
        ["sample_id", "source_id", "population", "cond", "generator", "split", "video_path"],
    )
    if args.limit and args.limit > 0:
        in_rows = in_rows[:args.limit]

    devices = parse_devices(args.device, args.devices)
    out_manifest = Path(args.out_manifest)
    fail_path = out_manifest.with_suffix(".failures.csv")

    if len(devices) == 1:
        args.device = devices[0]
        feature_rows, failures = extract_rows(args, in_rows)
    else:
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        shards = [in_rows[i::len(devices)] for i in range(len(devices))]
        ctx = mp.get_context("spawn")
        jobs = []
        shard_paths = []
        for worker_id, (device, shard) in enumerate(zip(devices, shards)):
            success_csv = out_manifest.with_name(f"{out_manifest.stem}.worker{worker_id}.csv")
            failure_csv = out_manifest.with_name(f"{out_manifest.stem}.worker{worker_id}.failures.csv")
            for stale in (success_csv, failure_csv):
                if stale.exists():
                    stale.unlink()
            shard_paths.append((success_csv, failure_csv))
            job = ctx.Process(
                target=worker_main,
                args=(args, shard, device, worker_id, str(success_csv), str(failure_csv)),
            )
            job.start()
            jobs.append(job)
        for job in jobs:
            job.join()
        failed_jobs = [job.pid for job in jobs if job.exitcode != 0]
        if failed_jobs:
            raise RuntimeError(f"id_audio feature workers failed: pids={failed_jobs}")
        feature_rows, failures = [], []
        for success_csv, failure_csv in shard_paths:
            if success_csv.exists():
                feature_rows.extend(read_csv(success_csv))
            if failure_csv.exists():
                failures.extend(read_csv(failure_csv))

    write_csv(out_manifest, feature_rows)
    if failures:
        write_csv(fail_path, failures)
        print(f"[id-audio-extract] failures -> {fail_path}")
    elif fail_path.exists():
        fail_path.unlink()
    print(f"[id-audio-extract] feature manifest -> {out_manifest} ok={len(feature_rows)} failed={len(failures)}")


if __name__ == "__main__":
    main()
