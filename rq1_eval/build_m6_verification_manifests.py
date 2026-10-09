#!/usr/bin/env python3
"""Build combined feature manifests for M6 media-realized verification.

M6 extracts features only for newly realized media. This helper merges those
new rows with frozen enrollment originals and L1 calibration rows from prior
M1/M2/M3 artifacts so downstream evaluators can reuse calibrated thresholds
without re-extracting hundreds of baseline clips.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

M6_META_COLUMNS = [
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
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[m6-build-manifests] rows={len(df)} -> {path}")


def dedupe_by_sample(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    seen = set()
    for frame in frames:
        if frame.empty:
            continue
        if "sample_id" not in frame.columns:
            raise ValueError("feature manifest missing sample_id")
        keep = []
        for _, row in frame.iterrows():
            sid = str(row.get("sample_id", ""))
            if not sid or sid in seen:
                continue
            seen.add(sid)
            keep.append(row)
        if keep:
            rows.append(pd.DataFrame(keep))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).fillna("")


def dedupe_orig_by_enroll(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep a single enrollment original per enroll_id.

    Combined manifests may include both frozen baseline originals and newly
    extracted clean-source originals. Downstream evaluators expect each
    enroll_id to map to one registered original sample. Preserve row order so
    frozen baseline rows win over supplemental clean-source rows.
    """
    if frame.empty or "population" not in frame.columns or "enroll_id" not in frame.columns:
        return frame
    seen_orig_enrolls = set()
    keep = []
    for idx, row in frame.iterrows():
        is_orig = str(row.get("population", "")) == "orig"
        enroll_id = str(row.get("enroll_id", ""))
        if is_orig and enroll_id:
            if enroll_id in seen_orig_enrolls:
                continue
            seen_orig_enrolls.add(enroll_id)
        keep.append(idx)
    return frame.loc[keep].reset_index(drop=True).fillna("")


def enrich_completed_meta(frame: pd.DataFrame, completed: pd.DataFrame) -> pd.DataFrame:
    """Backfill M6 metadata lost by feature extractors."""
    if frame.empty or completed.empty or "sample_id" not in frame.columns:
        return frame
    completed_by_sample = completed.drop_duplicates("sample_id").set_index("sample_id", drop=False)
    out = frame.copy()
    for col in M6_META_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    for idx, row in out.iterrows():
        sid = str(row.get("sample_id", ""))
        if sid not in completed_by_sample.index:
            continue
        meta = completed_by_sample.loc[sid]
        for col in M6_META_COLUMNS:
            current = str(out.at[idx, col]) if col in out.columns else ""
            if current and current.lower() != "nan":
                continue
            value = str(meta.get(col, ""))
            if value and value.lower() != "nan":
                out.at[idx, col] = value
    return out.fillna("")


def selected_baseline_rows(paths: Sequence[Path], needed_enroll_ids: set[str]) -> pd.DataFrame:
    parts = []
    for path in paths:
        frame = read_csv(path)
        if frame.empty:
            continue
        for col in ["population", "cond", "split", "enroll_id"]:
            if col not in frame.columns:
                frame[col] = ""
        l1_calib = frame[
            (frame["population"].astype(str) == "same")
            & (frame["cond"].astype(str) == "L1")
            & (frame["split"].astype(str) == "calib")
        ]
        required_orig_enroll_ids = set(needed_enroll_ids) | set(l1_calib["enroll_id"].astype(str))
        orig = frame[
            (frame["population"].astype(str) == "orig")
            & frame["enroll_id"].astype(str).isin(required_orig_enroll_ids)
        ]
        parts.extend([orig, l1_calib])
    return dedupe_by_sample(parts)


def require_paths(df: pd.DataFrame, path_cols: Sequence[str], label: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"{label}: no rows after merge")
    missing = [col for col in path_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing path columns {missing}")
    # Keep rows where at least one required path is available. Some video rows
    # intentionally keep partial channel availability for fail-safe accounting.
    mask = False
    for col in path_cols:
        mask = mask | df[col].astype(str).ne("")
    out = df[mask].copy()
    if out.empty:
        raise ValueError(f"{label}: no rows with feature paths")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--completed-inventory", type=Path, required=True)
    ap.add_argument("--m6-id-audio", type=Path, required=True)
    ap.add_argument("--m6-video", type=Path, required=True)
    ap.add_argument("--m6-video-window", type=Path, required=True)
    ap.add_argument("--m6-evt-audio", type=Path, required=True)
    ap.add_argument("--id-audio-baseline", type=Path, action="append", default=[])
    ap.add_argument("--video-baseline", type=Path, action="append", default=[])
    ap.add_argument("--video-window-baseline", type=Path, action="append", default=[])
    ap.add_argument("--evt-audio-baseline", type=Path, action="append", default=[])
    ap.add_argument("--out-id-audio", type=Path, required=True)
    ap.add_argument("--out-video", type=Path, required=True)
    ap.add_argument("--out-video-window", type=Path, required=True)
    ap.add_argument("--out-evt-audio", type=Path, required=True)
    args = ap.parse_args()

    completed = read_csv(args.completed_inventory)
    if completed.empty:
        raise ValueError("completed inventory is empty")
    needed_enroll_ids = set(str(x) for x in completed["enroll_id"].dropna().astype(str))

    id_audio = enrich_completed_meta(dedupe_by_sample([
        read_csv(args.m6_id_audio),
        selected_baseline_rows(args.id_audio_baseline, needed_enroll_ids),
    ]), completed)
    video = enrich_completed_meta(dedupe_by_sample([
        read_csv(args.m6_video),
        selected_baseline_rows(args.video_baseline, needed_enroll_ids),
    ]), completed)
    video_window = enrich_completed_meta(dedupe_by_sample([
        read_csv(args.m6_video_window),
        selected_baseline_rows(args.video_window_baseline, needed_enroll_ids),
    ]), completed)
    evt_audio = enrich_completed_meta(dedupe_by_sample([
        read_csv(args.m6_evt_audio),
        selected_baseline_rows(args.evt_audio_baseline, needed_enroll_ids),
    ]), completed)

    id_audio = dedupe_orig_by_enroll(id_audio)
    video = dedupe_orig_by_enroll(video)
    video_window = dedupe_orig_by_enroll(video_window)
    evt_audio = dedupe_orig_by_enroll(evt_audio)

    id_audio = require_paths(id_audio, ["id_audio_path"], "id_audio")
    video = require_paths(video, ["id_video_path", "evt_video_path"], "video")
    video_window = require_paths(video_window, ["window_feature_path"], "video_window")
    evt_audio = require_paths(evt_audio, ["evt_audio_windows_path"], "evt_audio")

    write_csv(args.out_id_audio, id_audio)
    write_csv(args.out_video, video)
    write_csv(args.out_video_window, video_window)
    write_csv(args.out_evt_audio, evt_audio)


if __name__ == "__main__":
    main()
