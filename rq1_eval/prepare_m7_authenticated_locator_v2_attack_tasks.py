#!/usr/bin/env python3
"""Rewrite N1000 attack tasks onto authenticated locator v2 media.

Resolution is keyed by the previous signed registry record hash.  This avoids
the enroll-id overwrite that occurs when one claim family has multiple signed
versions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def unique_index(rows: Sequence[Dict[str, str]], key: str, label: str) -> Dict[str, Dict[str, str]]:
    index: Dict[str, Dict[str, str]] = {}
    for row in rows:
        value = str(row.get(key, "")).strip()
        if not value:
            raise ValueError(f"{label} row is missing {key}")
        if value in index:
            raise ValueError(f"duplicate {label} key {value}")
        index[value] = row
    return index


def optional_unique_index(
    rows: Sequence[Dict[str, str]], key: str
) -> Dict[str, Dict[str, str]]:
    index: Dict[str, Dict[str, str]] = {}
    duplicates = set()
    for row in rows:
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        if value in index:
            duplicates.add(value)
        else:
            index[value] = row
    for value in duplicates:
        index.pop(value, None)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, action="append", required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--embedded-manifest", type=Path, action="append", required=True)
    parser.add_argument("--signed-registry", type=Path, required=True)
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out-tasks", type=Path, required=True)
    parser.add_argument("--out-missing", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()

    mapping_rows = read_csv(args.mapping)
    mapping = unique_index(
        mapping_rows, "previous_registry_record_sha256", "locator mapping"
    )
    mapping_by_sample = optional_unique_index(mapping_rows, "sample_id")
    mapping_by_enroll = optional_unique_index(mapping_rows, "enroll_id")
    embedded_rows: List[Dict[str, str]] = []
    for path in args.embedded_manifest:
        embedded_rows.extend(read_csv(path))
    embedded = unique_index(
        embedded_rows, "m7_payload_registry_record_sha256", "embedded manifest"
    )

    source_tasks: List[Dict[str, str]] = []
    for path in args.tasks:
        source_tasks.extend(read_csv(path))

    rewritten: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    seen_task_ids = set()
    for task in source_tasks:
        old_hash = str(task.get("m7_payload_registry_record_sha256", "")).strip()
        migrated = mapping.get(old_hash)
        if migrated is None:
            sample_id = str(task.get("sample_id", "")).strip()
            enroll_id = str(task.get("enroll_id", "")).strip()
            migrated = mapping_by_sample.get(sample_id) or mapping_by_enroll.get(enroll_id)
        if migrated is None:
            missing.append({**task, "missing_reason": "claim_record_not_mapped"})
            continue
        old_hash = str(migrated["previous_registry_record_sha256"])
        new_hash = str(migrated["registry_record_sha256"])
        carrier = embedded.get(new_hash)
        if carrier is None:
            missing.append({**task, "missing_reason": "v2_embedded_media_not_found"})
            continue
        if str(carrier.get("status", "")) != "ok":
            missing.append({**task, "missing_reason": "v2_embedded_media_not_ok"})
            continue

        old_task_id = str(task.get("m6_original_task_id") or task.get("task_id") or task.get("sample_id"))
        task_id = f"{old_task_id}__M7locatorv2"
        if task_id in seen_task_ids:
            raise ValueError(f"duplicate rewritten task id {task_id}")
        seen_task_ids.add(task_id)
        attack_type = str(task.get("m6_attack_type") or task.get("cond") or "attack")
        generator = str(task.get("generator") or "generator")
        output = args.output_root / slug(attack_type) / slug(generator) / f"{slug(task_id)}.mp4"
        candidate_source = str(task.get("m6_candidate_source", "")).strip()
        if "m7_authenticated_locator_v2_embedded_input" not in candidate_source:
            candidate_source = (
                f"{candidate_source}|m7_authenticated_locator_v2_embedded_input"
                if candidate_source
                else "m7_authenticated_locator_v2_embedded_input"
            )

        row: Dict[str, Any] = dict(task)
        row.update(
            {
                "task_id": task_id,
                "sample_id": task_id,
                "input_video_path": carrier["embedded_video_path"],
                "output_video_path": str(output),
                "m6_original_task_id": old_task_id,
                "m6_candidate_source": candidate_source,
                "status": "pending",
                "m7_carrier_policy": "m7-authenticated-claim-locator-v2",
                "m7_audio_carrier": "audio32_authenticated_claim_locator_v2",
                "m7_video_carrier": "videoseal_repeat24_authenticated_claim_locator_v2",
                "m7_original_input_video_path": carrier.get("original_video_path", ""),
                "m7_embedded_input_video_path": carrier["embedded_video_path"],
                "m7_previous_registry_record_sha256": old_hash,
                "m7_payload_registry_record_sha256": new_hash,
                "m7_payload_signed_registry": str(args.signed_registry),
                "m7_payload_public_key": str(args.public_key),
                "authenticated_claim_locator_hex": migrated.get(
                    "authenticated_claim_locator_hex", ""
                ),
                "claim_family_sha256": migrated.get("claim_family_sha256", ""),
                "claim_version": migrated.get("claim_version", ""),
            }
        )
        rewritten.append(row)

    write_csv(args.out_tasks, rewritten)
    write_csv(args.out_missing, missing)
    by_generator = Counter(str(row.get("generator", "")) for row in rewritten)
    by_attack = Counter(str(row.get("m6_attack_type", "")) for row in rewritten)
    summary = {
        "source_task_files": [str(path) for path in args.tasks],
        "source_tasks": len(source_tasks),
        "rewritten_tasks": len(rewritten),
        "missing_tasks": len(missing),
        "unique_v2_records_used": len(
            {str(row["m7_payload_registry_record_sha256"]) for row in rewritten}
        ),
        "by_generator": dict(sorted(by_generator.items())),
        "by_attack_type": dict(sorted(by_attack.items())),
        "mapping": str(args.mapping),
        "embedded_manifests": [str(path) for path in args.embedded_manifest],
        "signed_registry": str(args.signed_registry),
        "public_key": str(args.public_key),
        "output_root": str(args.output_root),
        "out_tasks": str(args.out_tasks),
        "out_missing": str(args.out_missing),
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if missing or len(rewritten) != len(source_tasks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
