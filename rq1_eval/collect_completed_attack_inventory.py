#!/usr/bin/env python3
"""Collect completed external-generation tasks into an RQ1 attack inventory."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


OUTPUT_COLUMNS = [
    "task_id", "sample_id", "enroll_id", "source_id", "population", "cond", "generator",
    "split", "video_path", "attack_level", "attack_family", "attack_scope",
    "target_start_sec", "target_end_sec", "generator_role",
    "attack_variant", "splice_duration_sec", "splice_position",
    "codec_bandwidth", "enhancement_model", "attack_language",
    "video_crf", "audio_bitrate",
    "m6_attack_type", "m6_candidate_policy", "m6_candidate_source",
    "m6_target_sample_id", "m6_donor_sample_id", "m6_target_source_id",
    "m6_donor_source_id", "m6_budget_bits", "m6_bits_needed_to_pass",
    "m6_original_task_id", "m6_realization_status",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--generators", nargs="+",
                    help="collect only these generators, e.g. ffmpeg_splice XTTS")
    ap.add_argument("--sample-id-col", default="sample_id",
                    help="source column to write as output sample_id")
    ap.add_argument("--allow-missing", action="store_true",
                    help="write completed rows and report pending rows instead of failing")
    args = ap.parse_args()

    rows = read_csv(args.tasks)
    selected = set(args.generators or [])
    if selected:
        rows = [row for row in rows if row["generator"] in selected]
    completed = []
    missing = []
    for row in rows:
        output = Path(row["output_video_path"])
        if not output.exists():
            missing.append(row)
            continue
        rec = {col: row.get(col, "") for col in OUTPUT_COLUMNS}
        if args.sample_id_col not in row:
            raise ValueError(f"task CSV missing --sample-id-col {args.sample_id_col!r}")
        rec["sample_id"] = row.get(args.sample_id_col, "")
        if not rec["sample_id"]:
            raise ValueError(f"row has empty --sample-id-col {args.sample_id_col!r}")
        rec["video_path"] = str(output)
        completed.append(rec)
    if missing and not args.allow_missing:
        raise SystemExit(
            f"{len(missing)} / {len(rows)} attack outputs are missing; "
            "rerun generators or pass --allow-missing for a partial pilot"
        )
    write_csv(args.out, completed)
    print(f"[attack-collect] completed={len(completed)} missing={len(missing)} -> {args.out}")
    by = {}
    for row in missing:
        by[row["generator"]] = by.get(row["generator"], 0) + 1
    for generator, count in sorted(by.items()):
        print(f"  pending {generator}: {count}")


if __name__ == "__main__":
    main()
