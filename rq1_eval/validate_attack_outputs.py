#!/usr/bin/env python3
"""Validate generated attack videos with ffprobe before feature extraction.

The collector checks that files exist. This script performs the stricter media
check needed before a server pilot enters the expensive semantic encoders:

- output file is probeable;
- at least one video stream and one audio stream exist;
- duration is non-trivial;
- when an input video path is available, output duration stays close to input.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def probe(ffprobe: str, path: Path) -> Dict[str, Union[float, int]]:
    if not path.exists():
        raise FileNotFoundError(path)
    cmd = [
        ffprobe, "-v", "error", "-show_entries",
        "format=duration:stream=codec_type", "-of", "json", str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)
    streams = data.get("streams", [])
    return {
        "duration_sec": duration,
        "video_streams": sum(stream.get("codec_type") == "video" for stream in streams),
        "audio_streams": sum(stream.get("codec_type") == "audio" for stream in streams),
    }


def choose_path_col(rows: List[Dict[str, str]], requested: Optional[str]) -> str:
    if requested:
        return requested
    for candidate in ("video_path", "output_video_path"):
        if candidate in rows[0]:
            return candidate
    raise ValueError("CSV has neither video_path nor output_video_path; pass --path-col")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True,
                    help="attack inventory or task CSV")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--path-col", default=None,
                    help="defaults to video_path, then output_video_path")
    ap.add_argument("--input-path-col", default="input_video_path",
                    help="optional original-video column for duration-ratio checks")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--min-duration-sec", type=float, default=0.5)
    ap.add_argument("--min-duration-ratio", type=float, default=0.90)
    ap.add_argument("--max-duration-ratio", type=float, default=1.10)
    ap.add_argument("--allow-failed", action="store_true")
    args = ap.parse_args()

    rows = read_csv(args.csv)
    if not rows:
        raise ValueError("input CSV is empty")
    path_col = choose_path_col(rows, args.path_col)
    if path_col not in rows[0]:
        raise ValueError(f"CSV missing path column {path_col!r}")
    has_input = args.input_path_col in rows[0]

    reports = []
    failures = 0
    for index, row in enumerate(rows, start=1):
        rec = dict(row)
        rec["checked_path"] = row.get(path_col, "")
        try:
            output = probe(args.ffprobe, Path(rec["checked_path"]))
            rec.update({f"output_{key}": value for key, value in output.items()})
            reasons = []
            if output["video_streams"] < 1:
                reasons.append("missing video stream")
            if output["audio_streams"] < 1:
                reasons.append("missing audio stream")
            if output["duration_sec"] < args.min_duration_sec:
                reasons.append(f"duration<{args.min_duration_sec}")
            if has_input and row.get(args.input_path_col):
                original = probe(args.ffprobe, Path(row[args.input_path_col]))
                rec["input_duration_sec"] = original["duration_sec"]
                ratio = (
                    output["duration_sec"] / original["duration_sec"]
                    if original["duration_sec"] > 0 else float("nan")
                )
                rec["duration_ratio"] = ratio
                if not (args.min_duration_ratio <= ratio <= args.max_duration_ratio):
                    reasons.append(
                        f"duration_ratio outside "
                        f"[{args.min_duration_ratio},{args.max_duration_ratio}]"
                    )
            if reasons:
                raise ValueError("; ".join(reasons))
            rec["validation_status"] = "ok"
            rec["validation_error"] = ""
            print(f"[validate {index}/{len(rows)}] ok     {rec['checked_path']}", flush=True)
        except Exception as exc:
            failures += 1
            rec["validation_status"] = "failed"
            rec["validation_error"] = str(exc)[-1000:]
            print(f"[validate {index}/{len(rows)}] FAILED {rec['checked_path']}: {exc}", flush=True)
        reports.append(rec)

    write_csv(args.out, reports)
    print(f"[validate] rows={len(reports)} failed={failures} -> {args.out}")
    if failures and not args.allow_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
