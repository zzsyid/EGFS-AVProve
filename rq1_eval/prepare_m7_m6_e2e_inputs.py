#!/usr/bin/env python3
"""Prepare M7-short-pointer inputs for M6 media-realized attacks.

This helper has two jobs:

1. Build a clean manifest for the unique M6 target sources that need an M7
   carrier claim.
2. Rewrite M6 attack tasks so their target input is the M7 embedded media while
   the enrolled M4 claim (`enroll_id`, `source_id`) stays unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def slug(text: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "item"


def parse_generators(text: str) -> set[str]:
    return {part.strip() for part in str(text).split(",") if part.strip()}


def select_tasks(
    rows: Sequence[Dict[str, str]],
    generators: set[str],
    limit: int,
    per_generator_limit: int = 0,
    splits: set[str] | None = None,
) -> List[Dict[str, str]]:
    splits = splits or set()
    selected = [
        row
        for row in rows
        if (not generators or row.get("generator", "") in generators)
        and (not splits or str(row.get("split", "")).strip() in splits)
    ]
    if per_generator_limit > 0:
        counts: Dict[str, int] = {}
        balanced: List[Dict[str, str]] = []
        for row in selected:
            generator = row.get("generator", "")
            if counts.get(generator, 0) >= per_generator_limit:
                continue
            balanced.append(row)
            counts[generator] = counts.get(generator, 0) + 1
        selected = balanced
    if limit > 0:
        selected = selected[:limit]
    return [dict(row) for row in selected]


def m7_split_for_m6(split: str) -> str:
    split = str(split).strip().lower()
    if split == "train":
        return "train"
    if split in {"calib", "val", "validation"}:
        return "val"
    return "test"


def build_clean_manifest(tasks: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    seen: Dict[str, str] = {}
    for task in tasks:
        enroll_id = str(task.get("enroll_id", "")).strip()
        video_path = str(task.get("input_video_path", "")).strip()
        if not enroll_id or not video_path:
            continue
        previous = seen.get(enroll_id)
        if previous and previous != video_path:
            raise ValueError(
                f"enroll_id {enroll_id!r} maps to multiple M6 input paths: "
                f"{previous!r} vs {video_path!r}"
            )
        if previous:
            continue
        seen[enroll_id] = video_path
        rows.append({
            "sample_id": enroll_id,
            "enroll_id": enroll_id,
            "source_id": task.get("source_id", ""),
            "split": task.get("split", ""),
            "original_split": task.get("split", ""),
            "m7_split": m7_split_for_m6(task.get("split", "")),
            "population": "orig",
            "cond": "clean",
            "generator": "none",
            "video_path": video_path,
            "m6_seed_task_id": task.get("task_id", ""),
        })
    return rows


def embedded_by_enroll(path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv(path)
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        enroll_id = str(row.get("enroll_id", "")).strip()
        embedded = str(row.get("embedded_video_path", "")).strip()
        if enroll_id and embedded:
            out[enroll_id] = row
    return out


def rewrite_tasks(
    tasks: Sequence[Dict[str, str]],
    embedded_manifest: Path,
    output_root: Path,
    suffix: str,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    embedded = embedded_by_enroll(embedded_manifest)
    rewritten: List[Dict[str, object]] = []
    missing: List[Dict[str, object]] = []
    for task in tasks:
        enroll_id = str(task.get("enroll_id", "")).strip()
        carrier = embedded.get(enroll_id)
        if carrier is None:
            missing.append({**task, "missing_reason": "no_m7_embedded_media_for_enroll_id"})
            continue
        old_task_id = str(task.get("task_id") or task.get("sample_id"))
        new_task_id = f"{old_task_id}__{suffix}"
        attack_type = str(task.get("m6_attack_type") or task.get("cond") or "m6_attack")
        generator = str(task.get("generator", "generator"))
        candidate_source = str(task.get("m6_candidate_source", "")).strip()
        if not candidate_source:
            candidate_source = str(task.get("generator") or task.get("attack_family") or "task_manifest")
        row: Dict[str, object] = dict(task)
        row.update({
            "task_id": new_task_id,
            "sample_id": new_task_id,
            "input_video_path": carrier["embedded_video_path"],
            "output_video_path": str(output_root / slug(attack_type) / slug(generator) / f"{slug(new_task_id)}.mp4"),
            "m6_attack_type": task.get("m6_attack_type") or attack_type,
            "m6_candidate_policy": task.get("m6_candidate_policy") or "m7_media_replay",
            "m6_original_task_id": task.get("m6_original_task_id") or old_task_id,
            "m6_candidate_source": f"{candidate_source}|m7_short_pointer_embedded_input",
            "m6_target_sample_id": task.get("m6_target_sample_id") or task.get("enroll_id") or "",
            "m6_donor_sample_id": task.get("m6_donor_sample_id") or "",
            "m6_target_source_id": task.get("m6_target_source_id") or task.get("source_id") or "",
            "m6_donor_source_id": task.get("m6_donor_source_id") or "",
            "m6_realization_status": task.get("m6_realization_status") or "media_realized",
            "m7_carrier_policy": "m7-short-pointer-v1",
            "m7_audio_carrier": "audio32_packet_beacon_v1",
            "m7_video_carrier": "videoseal_repeat24_short_pointer",
            "m7_original_input_video_path": task.get("input_video_path", ""),
            "m7_embedded_input_video_path": carrier["embedded_video_path"],
            "m7_payload_registry_record_sha256": carrier.get("m7_payload_registry_record_sha256", ""),
            "m7_payload_signed_registry": carrier.get("signed_payload_registry", ""),
            "m7_payload_public_key": carrier.get("public_key", ""),
        })
        rewritten.append(row)
    return rewritten, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m6-tasks", type=Path, required=True)
    parser.add_argument("--generators", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-generator-limit", type=int, default=0)
    parser.add_argument("--splits", default="")
    parser.add_argument("--out-clean-manifest", type=Path)
    parser.add_argument("--out-selected-tasks", type=Path)
    parser.add_argument("--embedded-manifest", type=Path)
    parser.add_argument("--out-rewritten-tasks", type=Path)
    parser.add_argument("--out-missing", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("/path/to/datasets/rq1_m7_m6_media_adaptive"))
    parser.add_argument("--task-suffix", default="M7shortptr")
    parser.add_argument("--out-summary", type=Path)
    args = parser.parse_args()

    tasks = select_tasks(
        read_csv(args.m6_tasks),
        parse_generators(args.generators),
        args.limit,
        args.per_generator_limit,
        parse_generators(args.splits),
    )
    summary: Dict[str, object] = {
        "m6_tasks": str(args.m6_tasks),
        "selected_tasks": len(tasks),
        "generators": sorted(parse_generators(args.generators)),
        "limit": args.limit,
        "per_generator_limit": args.per_generator_limit,
        "splits": sorted(parse_generators(args.splits)),
    }

    if args.out_clean_manifest:
        clean = build_clean_manifest(tasks)
        write_csv(args.out_clean_manifest, clean)
        summary["clean_manifest"] = str(args.out_clean_manifest)
        summary["clean_records"] = len(clean)

    if args.out_selected_tasks:
        write_csv(args.out_selected_tasks, tasks)
        summary["selected_tasks_manifest"] = str(args.out_selected_tasks)

    if args.embedded_manifest and args.out_rewritten_tasks:
        rewritten, missing = rewrite_tasks(tasks, args.embedded_manifest, args.output_root, args.task_suffix)
        write_csv(args.out_rewritten_tasks, rewritten)
        if args.out_missing:
            write_csv(args.out_missing, missing)
        summary.update({
            "embedded_manifest": str(args.embedded_manifest),
            "rewritten_tasks": str(args.out_rewritten_tasks),
            "rewritten_task_rows": len(rewritten),
            "missing_embedded_rows": len(missing),
            "output_root": str(args.output_root),
        })

    if args.out_summary:
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
