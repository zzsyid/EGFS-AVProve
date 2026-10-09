#!/usr/bin/env python3
"""Summarize sharded authenticated-locator-v2 N1000 verification outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
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


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def metrics(rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    n = len(rows)

    def count(field: str) -> int:
        return sum(int(truth(row.get(field, ""))) for row in rows)

    resolved = count("claim_resolved_authenticated")
    expected = count("claim_expected_hit")
    final_ok = count("claim_final_ok")
    wrong = count("claim_wrong_source")
    audio = count("audio_tag_recovered")
    video = count("video_pointer_recovered")
    return {
        "n": n,
        "authenticated_resolved": resolved,
        "authenticated_resolved_rate": resolved / n if n else None,
        "expected_record_hit": expected,
        "expected_record_hit_rate": expected / n if n else None,
        "claim_final_ok": final_ok,
        "claim_final_ok_rate": final_ok / n if n else None,
        "wrong_source_selected": wrong,
        "wrong_source_rate": wrong / n if n else None,
        "unresolved": n - resolved,
        "unresolved_rate": (n - resolved) / n if n else None,
        "audio_tag_recovered": audio,
        "audio_tag_recovery_rate": audio / n if n else None,
        "video_pointer_recovered": video,
        "video_pointer_recovery_rate": video / n if n else None,
    }


def grouped(rows: Sequence[Dict[str, str]], field: str) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(field, ""))].append(row)
    return [{field: key, **metrics(value)} for key, value in sorted(buckets.items())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, action="append", required=True)
    parser.add_argument("--failures", type=Path, action="append", default=[])
    parser.add_argument("--expected-n", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: List[Dict[str, str]] = []
    for path in args.rows:
        rows.extend(read_csv(path))
    failures: List[Dict[str, str]] = []
    for path in args.failures:
        failures.extend(read_csv(path))

    task_ids = [str(row.get("task_id", "")) for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate task IDs across verification shards")
    if len(rows) + len(failures) != args.expected_n:
        raise ValueError(
            f"verification closure mismatch: rows={len(rows)} failures={len(failures)} "
            f"expected={args.expected_n}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined_rows = args.out_dir / "n1000_authenticated_locator_v2_claim_rows.csv"
    by_generator_path = args.out_dir / "n1000_authenticated_locator_v2_by_generator.csv"
    by_attack_path = args.out_dir / "n1000_authenticated_locator_v2_by_attack_type.csv"
    failures_path = args.out_dir / "n1000_authenticated_locator_v2_failures.csv"
    summary_path = args.out_dir / "n1000_authenticated_locator_v2_summary.json"
    write_csv(combined_rows, rows)
    write_csv(by_generator_path, grouped(rows, "generator"))
    write_csv(by_attack_path, grouped(rows, "m6_attack_type"))
    write_csv(failures_path, failures)
    summary = {
        "policy": "tdsc-m7-authenticated-claim-locator-v2",
        "oracle_free_resolution": True,
        "expected_hash_used_for_selection": False,
        "expected_n": args.expected_n,
        "closed_n": len(rows) + len(failures),
        "verification_failures": len(failures),
        "overall": metrics(rows),
        "by_generator": grouped(rows, "generator"),
        "by_attack_type": grouped(rows, "m6_attack_type"),
        "inputs": {
            "rows": [str(path) for path in args.rows],
            "failures": [str(path) for path in args.failures],
        },
        "outputs": {
            "combined_rows": str(combined_rows),
            "by_generator": str(by_generator_path),
            "by_attack_type": str(by_attack_path),
            "failures": str(failures_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
