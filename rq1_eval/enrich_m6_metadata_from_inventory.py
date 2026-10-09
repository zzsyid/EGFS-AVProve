#!/usr/bin/env python3
"""Backfill M6/M7 metadata into a CSV by sample_id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_COLUMNS = [
    "m6_attack_type",
    "m6_candidate_policy",
    "m6_candidate_source",
    "m6_target_sample_id",
    "m6_donor_sample_id",
    "m6_target_source_id",
    "m6_donor_source_id",
    "m6_budget_bits",
    "m6_bits_needed_to_pass",
    "m6_original_task_id",
    "m6_realization_status",
    "m7_carrier_policy",
    "m7_audio_carrier",
    "m7_video_carrier",
    "m7_original_input_video_path",
    "m7_embedded_input_video_path",
    "m7_payload_registry_record_sha256",
    "m7_payload_signed_registry",
    "m7_payload_public_key",
]


def has_value(value: object) -> bool:
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--columns", default=",".join(DEFAULT_COLUMNS))
    parser.add_argument("--sample-id-col", default="sample_id")
    args = parser.parse_args()

    df = pd.read_csv(args.input).fillna("")
    inv = pd.read_csv(args.inventory).fillna("")
    if args.sample_id_col not in df.columns:
        raise ValueError(f"input missing {args.sample_id_col!r}")
    if args.sample_id_col not in inv.columns:
        raise ValueError(f"inventory missing {args.sample_id_col!r}")
    columns: List[str] = [col.strip() for col in args.columns.split(",") if col.strip()]
    inv_by_sample = inv.drop_duplicates(args.sample_id_col).set_index(args.sample_id_col, drop=False)
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    matched = 0
    filled = 0
    for idx, row in out.iterrows():
        sid = str(row.get(args.sample_id_col, ""))
        if sid not in inv_by_sample.index:
            continue
        matched += 1
        meta = inv_by_sample.loc[sid]
        for col in columns:
            if col not in inv.columns:
                continue
            if has_value(out.at[idx, col]):
                continue
            value = meta.get(col, "")
            if has_value(value):
                out.at[idx, col] = value
                filled += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    summary = {
        "input": str(args.input),
        "inventory": str(args.inventory),
        "out": str(args.out),
        "rows": int(len(out)),
        "inventory_rows": int(len(inv)),
        "matched_rows": int(matched),
        "filled_cells": int(filled),
        "columns": columns,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
