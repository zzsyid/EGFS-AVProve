#!/usr/bin/env python3
"""Evaluate window-level M2 video evidence for local-splice alarm and localization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from distances import d_ham_bits
from epsilon_sem import clopper_pearson
from quantize import QuantizerBank


CHANNELS = ["id_video", "evt_video"]


def cp_block(events: int, n: int) -> Dict[str, object]:
    lo, hi = clopper_pearson(events, n)
    return {
        "events": int(events),
        "n": int(n),
        "point": events / n if n else None,
        "clopper_pearson95": [lo, hi],
    }


def overlap(start: float, end: float, target_start: float, target_end: float) -> bool:
    return min(end, target_end) > max(start, target_start)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-bits", type=int, default=255)
    parser.add_argument("--q-l1", type=float, default=0.99)
    parser.add_argument("--master-key", default="rq1-m2-video-window-stage0")
    parser.add_argument("--window-span-sec", type=float, default=1.0)
    args = parser.parse_args()

    df = pd.read_csv(args.feature_manifest).fillna("")
    orig = df[df.population == "orig"].set_index("enroll_id", drop=False)
    if len(orig) == 0:
        raise ValueError("window feature manifest has no enrolled originals")
    cache = {
        str(row.sample_id): load_npz(str(row.window_feature_path))
        for _, row in df.iterrows()
    }
    dims = {
        channel: int(next(iter(cache.values()))[channel].shape[1])
        for channel in CHANNELS
    }
    qbank = QuantizerBank(dims, args.n_bits, kind="lsh", master_key=args.master_key.encode())

    window_rows: List[Dict[str, object]] = []
    for _, row in df[df.population != "orig"].iterrows():
        if row.enroll_id not in orig.index:
            continue
        reference_id = str(orig.loc[row.enroll_id].sample_id)
        reference = cache[reference_id]
        current = cache[str(row.sample_id)]
        common = min(len(reference["starts_sec"]), len(current["starts_sec"]))
        for index in range(common):
            start = float(current["starts_sec"][index])
            rec = {
                **row.to_dict(),
                "reference_sample_id": reference_id,
                "window_index": index,
                "window_start_sec": start,
                "window_end_sec": start + args.window_span_sec,
            }
            for channel in CHANNELS:
                available = bool(
                    reference[f"{channel}_available"][index]
                    and current[f"{channel}_available"][index]
                )
                rec[f"{channel}_available"] = available
                if available:
                    reference_code = qbank.encode(channel, reference[channel][index])
                    current_code = qbank.encode(channel, current[channel][index])
                    rec[f"{channel}_ham_bits"] = d_ham_bits(reference_code, current_code)
                else:
                    rec[f"{channel}_ham_bits"] = None
            window_rows.append(rec)
    windows = pd.DataFrame(window_rows)
    if len(windows) == 0:
        raise ValueError("no aligned non-original windows available")

    sample_max_rows = []
    for sample_id, group in windows.groupby("sample_id"):
        rec = {"sample_id": sample_id}
        for channel in CHANNELS:
            available = group[group[f"{channel}_available"]]
            rec[f"{channel}_max_bits"] = (
                int(available[f"{channel}_ham_bits"].max()) if len(available) else None
            )
        sample_max_rows.append(rec)
    sample_max = pd.DataFrame(sample_max_rows).set_index("sample_id")

    thresholds = {}
    l1_calib_ids = set(
        df[
            (df.population == "same") & (df.cond == "L1") & (df.split == "calib")
        ].sample_id
    )
    for channel in CHANNELS:
        values = sample_max.loc[
            sample_max.index.intersection(l1_calib_ids), f"{channel}_max_bits"
        ].dropna()
        if len(values) == 0:
            raise ValueError(f"{channel}: no L1 calibration sample maxima")
        thresholds[channel] = int(math.ceil(float(np.quantile(values, args.q_l1))))

    decisions = []
    for sample_id, group in windows.groupby("sample_id"):
        meta = group.iloc[0]
        scored = group.copy()
        for channel in CHANNELS:
            scored[f"{channel}_alarm"] = (
                scored[f"{channel}_available"]
                & (scored[f"{channel}_ham_bits"] > thresholds[channel])
            )
            scored[f"{channel}_ratio"] = (
                scored[f"{channel}_ham_bits"].fillna(-1) / max(1, thresholds[channel])
            )
        scored["window_alarm"] = scored[[f"{channel}_alarm" for channel in CHANNELS]].any(axis=1)
        scored["window_score"] = scored[[f"{channel}_ratio" for channel in CHANNELS]].max(axis=1)
        top = scored.sort_values("window_score", ascending=False).iloc[0]
        target_start = float(meta.target_start_sec) if str(meta.target_start_sec) else None
        target_end = float(meta.target_end_sec) if str(meta.target_end_sec) else None
        suspicious = scored[scored.window_alarm]
        any_overlap = None
        top_overlap = None
        if target_start is not None and target_end is not None:
            any_overlap = any(
                overlap(float(row.window_start_sec), float(row.window_end_sec), target_start, target_end)
                for _, row in suspicious.iterrows()
            )
            top_overlap = overlap(
                float(top.window_start_sec), float(top.window_end_sec), target_start, target_end,
            )
        decisions.append({
            "sample_id": sample_id,
            "enroll_id": meta.enroll_id,
            "split": meta.split,
            "population": meta.population,
            "cond": meta.cond,
            "generator": meta.generator,
            "attack_family": meta.attack_family,
            "target_start_sec": target_start,
            "target_end_sec": target_end,
            "window_count": int(len(scored)),
            "alarmed_window_count": int(scored.window_alarm.sum()),
            "window_alarm": bool(scored.window_alarm.any()),
            "top_window_start_sec": float(top.window_start_sec),
            "top_window_end_sec": float(top.window_end_sec),
            "top_window_score": float(top.window_score),
            "any_alarm_overlaps_target": any_overlap,
            "top_window_overlaps_target": top_overlap,
        })
        windows.loc[scored.index, "window_alarm"] = scored.window_alarm
        windows.loc[scored.index, "window_score"] = scored.window_score

    decisions = pd.DataFrame(decisions)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(args.out_dir / "m2_video_window_scores.csv", index=False)
    decisions.to_csv(args.out_dir / "m2_video_window_decisions.csv", index=False)

    summary = {"thresholds_bits": thresholds, "by_condition": {}}
    for (split, cond, generator), group in decisions.groupby(["split", "cond", "generator"]):
        overlap_group = group[group.any_alarm_overlaps_target.notna()]
        summary["by_condition"][f"{split}::{cond}::{generator}"] = {
            "n": int(len(group)),
            "window_alarm": cp_block(int(group.window_alarm.sum()), int(len(group))),
            "any_alarm_overlaps_target": cp_block(
                int(overlap_group.any_alarm_overlaps_target.sum()), int(len(overlap_group))
            ),
            "top_window_overlaps_target": cp_block(
                int(overlap_group.top_window_overlaps_target.sum()), int(len(overlap_group))
            ),
        }
    splice = decisions[decisions.cond == "video_splice"]
    summary["l1_overall"] = {}
    for split in ("calib", "eval"):
        group = decisions[(decisions.split == split) & (decisions.cond == "L1")]
        summary["l1_overall"][split] = {
            "window_alarm": cp_block(int(group.window_alarm.sum()), int(len(group))),
        }
    summary["video_splice"] = {
        "n": int(len(splice)),
        "window_alarm": cp_block(int(splice.window_alarm.sum()), int(len(splice))),
        "any_alarm_overlaps_target": cp_block(
            int(splice.any_alarm_overlaps_target.sum()), int(len(splice))
        ),
        "top_window_overlaps_target": cp_block(
            int(splice.top_window_overlaps_target.sum()), int(len(splice))
        ),
    }
    (args.out_dir / "m2_video_window_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
