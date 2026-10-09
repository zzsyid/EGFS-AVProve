#!/usr/bin/env python3
"""Fuse M1/M2 evidence into the M3 fail-safe AV provenance states.

This is a deterministic Stage-0 fusion layer. It consumes existing per-channel
CSV outputs and does not re-extract features.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


STATE_VERIFIED = "source-and-content verified"
STATE_CONTENT = "content-unverified"
STATE_MODAL = "tampered/modal-replacement"
LEGACY_STATE_MODAL = "modal-replacement"
STATE_LOCAL = "localized-tamper"
STATE_STRIPPED = "stripped-suspicious"
STATE_REJECTED = "rejected"
LEGACY_STATE_REJECTED = "unknown"

FAIL_STATES = {STATE_CONTENT, STATE_MODAL, STATE_LOCAL}

META_COLS = [
    "sample_id",
    "enroll_id",
    "source_id",
    "split",
    "population",
    "cond",
    "generator",
    "attack_level",
    "attack_family",
    "attack_scope",
    "generator_role",
    "attack_variant",
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


def boolish(value: object) -> Optional[bool]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return None
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def first_nonempty(values: Iterable[object]) -> object:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        text = str(value)
        if text and text.lower() != "nan":
            return value
    return ""


def normalize_simple_state(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"verified", STATE_VERIFIED}:
        return STATE_VERIFIED
    if text in {"content-unverified", "content_unverified"}:
        return STATE_CONTENT
    if text in {LEGACY_STATE_MODAL, STATE_MODAL, "modal_replacement"}:
        return STATE_MODAL
    if text in {"localized-tamper", "localized_tamper"}:
        return STATE_LOCAL
    if text in {"stripped-suspicious", "stripped_suspicious"}:
        return STATE_STRIPPED
    if text in {STATE_REJECTED, LEGACY_STATE_REJECTED}:
        return STATE_REJECTED
    return STATE_REJECTED


def cp95(k: int, n: int) -> List[float]:
    """Clopper-Pearson 95% interval, using the project implementation if present."""
    if n <= 0:
        return [0.0, 1.0]
    try:
        from epsilon_sem import clopper_pearson  # type: ignore

        lo, hi = clopper_pearson(k, n)
        return [float(lo), float(hi)]
    except Exception:
        # Wilson fallback keeps the script dependency-light for quick local smoke.
        z = 1.959963984540054
        p = k / n
        denom = 1.0 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return [max(0.0, center - half), min(1.0, center + half)]


def load_csvs(paths: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path).fillna("")
        frame["evidence_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate_by_sample(df: pd.DataFrame, evidence_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["sample_id", *META_COLS, *evidence_cols])
    if "sample_id" not in df.columns:
        raise ValueError("input CSV is missing sample_id")
    for col in META_COLS:
        if col not in df.columns:
            df[col] = ""
    for col in evidence_cols:
        if col not in df.columns:
            df[col] = ""

    rows: List[Dict[str, object]] = []
    for sample_id, group in df.groupby("sample_id", sort=False):
        row: Dict[str, object] = {"sample_id": sample_id}
        for col in META_COLS:
            row[col] = first_nonempty(group[col].tolist())
        for col in evidence_cols:
            row[col] = first_nonempty(group[col].tolist())
        rows.append(row)
    return pd.DataFrame(rows)


def load_m1_fusion(paths: Sequence[Path]) -> pd.DataFrame:
    df = load_csvs(paths)
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["m1_audio_event_state"] = out.get("state_audio_only", "").map(normalize_simple_state)

    def audio_event_state_v1(row: pd.Series) -> str:
        long_state = normalize_simple_state(row.get("audio_long_state"))
        short_state = normalize_simple_state(row.get("audio_short_state"))
        states = [state for state in [long_state, short_state] if state != STATE_REJECTED]
        if not states:
            return normalize_simple_state(row.get("state_audio_only"))
        if STATE_CONTENT in states:
            return STATE_CONTENT
        if states.count(STATE_LOCAL) >= 2:
            return STATE_LOCAL
        if STATE_VERIFIED in states:
            return STATE_VERIFIED
        if STATE_LOCAL in states:
            return STATE_REJECTED
        return STATE_REJECTED

    out["m1_audio_event_state_v1"] = out.apply(audio_event_state_v1, axis=1)

    def sync_state(row: pd.Series) -> str:
        scoped_available = boolish(row.get("scoped_sync_available"))
        global_available = boolish(row.get("global_sync_available"))
        if scoped_available is not None:
            if boolish(row.get("scoped_sync_anomaly")):
                return STATE_MODAL
            return STATE_VERIFIED
        if global_available is not None:
            if boolish(row.get("global_sync_anomaly")):
                return STATE_MODAL
            return STATE_VERIFIED
        return STATE_REJECTED

    out["m1_sync_state"] = out.apply(sync_state, axis=1)

    def sync_state_v1(row: pd.Series) -> str:
        global_available = boolish(row.get("global_sync_available"))
        scoped_available = boolish(row.get("scoped_sync_available"))
        if global_available is None and scoped_available is None:
            return STATE_REJECTED
        global_bad = boolish(row.get("global_sync_anomaly")) is True
        scoped_bad = boolish(row.get("scoped_sync_anomaly")) is True
        if global_bad and scoped_bad:
            return STATE_MODAL
        return STATE_REJECTED

    out["m1_sync_state_v1"] = out.apply(sync_state_v1, axis=1)
    return aggregate_by_sample(
        out,
        ["m1_audio_event_state", "m1_audio_event_state_v1", "m1_sync_state", "m1_sync_state_v1"],
    )


def load_audio_event(paths: Sequence[Path], label: str) -> pd.DataFrame:
    df = load_csvs(paths)
    if df.empty:
        return pd.DataFrame()
    state_col = "state_p090" if "state_p090" in df.columns else "state"
    if state_col not in df.columns:
        raise ValueError(f"audio event CSV missing state column: {paths}")
    evidence_col = f"audio_event_{label}_state"
    df[evidence_col] = df[state_col].map(normalize_simple_state)
    return aggregate_by_sample(df, [evidence_col])


def load_audio_id(paths: Sequence[Path]) -> pd.DataFrame:
    df = load_csvs(paths)
    if df.empty:
        return pd.DataFrame()

    def state(row: pd.Series) -> str:
        available = boolish(row.get("id_audio_available"))
        verified = boolish(row.get("id_audio_verified"))
        if available is False or verified is None:
            return STATE_REJECTED
        if verified is True:
            return STATE_VERIFIED
        return STATE_MODAL

    df["audio_id_state"] = df.apply(state, axis=1)
    return aggregate_by_sample(df, ["audio_id_state"])


def load_video_whole(paths: Sequence[Path]) -> pd.DataFrame:
    df = load_csvs(paths)
    if df.empty:
        return pd.DataFrame()

    def state(row: pd.Series) -> str:
        feature_available = boolish(row.get("feature_available"))
        if feature_available is False:
            return STATE_REJECTED
        if boolish(row.get("video_verified")) is True:
            return STATE_VERIFIED
        if boolish(row.get("identity_consistent")) is False:
            return STATE_MODAL
        if boolish(row.get("visual_event_consistent")) is False:
            return STATE_CONTENT
        return STATE_REJECTED

    df["video_whole_state"] = df.apply(state, axis=1)

    def state_v1(row: pd.Series) -> str:
        base = state(row)
        if base == STATE_VERIFIED:
            return STATE_REJECTED
        return base

    df["video_whole_state_v1"] = df.apply(state_v1, axis=1)
    return aggregate_by_sample(df, ["video_whole_state", "video_whole_state_v1"])


def load_video_window(paths: Sequence[Path]) -> pd.DataFrame:
    df = load_csvs(paths)
    if df.empty:
        return pd.DataFrame()

    def state(row: pd.Series) -> str:
        alarm = boolish(row.get("window_alarm"))
        if alarm is True:
            return STATE_LOCAL
        if alarm is False:
            return STATE_VERIFIED
        return STATE_REJECTED

    df["video_window_state"] = df.apply(state, axis=1)

    def state_v1(row: pd.Series) -> str:
        return STATE_LOCAL if boolish(row.get("window_alarm")) is True else STATE_REJECTED

    df["video_window_state_v1"] = df.apply(state_v1, axis=1)
    return aggregate_by_sample(df, ["video_window_state", "video_window_state_v1"])


def load_audio_window_alarm(
    paths: Sequence[Path],
    long_run: int,
    short_run: int,
    max_failed_windows: int,
    ham_threshold: int,
    q90_threshold: float,
    compact_failed_ratio: float,
    compact_ham_threshold: int,
    coherent_run: int,
    coherent_failed_ratio: float,
    coherent_ham_threshold: int,
    diffuse_failed_ratio: float,
    diffuse_recovery_ratio: float,
    diffuse_min_failed_windows: int,
    global_failed_ratio: float,
    global_q90_threshold: float,
    global_ham_threshold: int,
    long_failed_ratio: float,
) -> pd.DataFrame:
    df = load_csvs(paths)
    if df.empty:
        return pd.DataFrame()
    required = [
        "longest_failed_run",
        "failed_windows",
        "max_ham_bits",
        "q90_ham_bits",
        "window_count",
        "recovery_ratio",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"audio-window CSV missing columns {missing}: {paths}")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    window_count = df["window_count"].replace(0, pd.NA)
    failed_ratio = (df["failed_windows"] / window_count).fillna(0.0)
    diffuse_degradation = (failed_ratio >= diffuse_failed_ratio) | (
        (df["recovery_ratio"] <= diffuse_recovery_ratio)
        & (df["failed_windows"] >= diffuse_min_failed_windows)
    )

    # Short, high-distance clusters are the typical signature of sub-second
    # audio splice residuals. Keep the original high-confidence short rule, and
    # add a slightly lower-Hamming compact rule for late 0.5s splices.
    alarm_short = (
        (df["longest_failed_run"] >= short_run)
        & (df["failed_windows"] <= max_failed_windows)
        & (df["max_ham_bits"] >= ham_threshold)
        & (df["q90_ham_bits"] <= q90_threshold)
    )
    alarm_compact = (
        (df["longest_failed_run"] >= short_run)
        & (df["failed_windows"] <= max_failed_windows)
        & (failed_ratio <= compact_failed_ratio)
        & (df["max_ham_bits"] >= compact_ham_threshold)
        & (df["q90_ham_bits"] <= q90_threshold)
    )
    alarm_coherent = (
        (df["longest_failed_run"] >= coherent_run)
        & (failed_ratio <= coherent_failed_ratio)
        & (df["max_ham_bits"] >= coherent_ham_threshold)
    )

    # XTTS-like whole-sample regeneration is globally bad, not benign diffuse
    # quality loss. Preserve it as a downgrade even though failed_ratio is high.
    alarm_global_replacement = (
        (failed_ratio >= global_failed_ratio)
        & (df["q90_ham_bits"] >= global_q90_threshold)
        & (df["max_ham_bits"] >= global_ham_threshold)
    )
    alarm_long = (
        (df["longest_failed_run"] >= long_run)
        & (failed_ratio <= long_failed_ratio)
        & (~diffuse_degradation)
    )
    df["audio_window_alarm_bool"] = (
        alarm_short | alarm_compact | alarm_coherent | alarm_global_replacement | alarm_long
    )
    for col in META_COLS:
        if col not in df.columns:
            df[col] = ""
    rows: List[Dict[str, object]] = []
    for sample_id, group in df.groupby("sample_id", sort=False):
        row: Dict[str, object] = {"sample_id": sample_id}
        for col in META_COLS:
            row[col] = first_nonempty(group[col].tolist())
        row["audio_window_alarm_state"] = (
            STATE_LOCAL if bool(group["audio_window_alarm_bool"].any()) else STATE_REJECTED
        )
        rows.append(row)
    return pd.DataFrame(rows)


def merge_evidence(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError("no evidence CSVs were provided")
    merged = frames[0].copy()
    for frame in frames[1:]:
        merged = merged.merge(frame, on="sample_id", how="outer", suffixes=("", "__new"))
        for col in META_COLS:
            new_col = f"{col}__new"
            if new_col in merged.columns:
                if col in merged.columns:
                    merged[col] = [
                        old if str(old) and str(old).lower() != "nan" else new
                        for old, new in zip(merged[col], merged[new_col])
                    ]
                else:
                    merged[col] = merged[new_col]
                merged.drop(columns=[new_col], inplace=True)
    for col in META_COLS:
        if col not in merged.columns:
            merged[col] = ""
    return merged.fillna("")


def fuse_branch(states: Sequence[object], claim_mode: str = "registered") -> str:
    normalized = [normalize_simple_state(state) for state in states if normalize_simple_state(state) != STATE_REJECTED]
    if not normalized:
        return STATE_REJECTED

    claim_recoverable = claim_mode == "registered"
    if STATE_MODAL in normalized:
        return STATE_MODAL if claim_recoverable else STATE_STRIPPED
    if STATE_LOCAL in normalized:
        return STATE_LOCAL if claim_recoverable else STATE_STRIPPED
    if STATE_CONTENT in normalized:
        return STATE_CONTENT if claim_recoverable else STATE_STRIPPED
    if all(state == STATE_VERIFIED for state in normalized):
        return STATE_VERIFIED if claim_recoverable else STATE_REJECTED
    return STATE_REJECTED


def fuse_audio_event_v1(row: pd.Series, claim_mode: str) -> str:
    m1_state = normalize_simple_state(row.get("m1_audio_event_state_v1"))
    if m1_state != STATE_REJECTED:
        states = [m1_state]
    else:
        states = [
            normalize_simple_state(row.get("audio_event_long_state")),
            normalize_simple_state(row.get("audio_event_short_state")),
        ]
        states = [state for state in states if state != STATE_REJECTED]
    if not states:
        return STATE_REJECTED
    if STATE_CONTENT in states:
        return STATE_CONTENT if claim_mode == "registered" else STATE_STRIPPED
    if states.count(STATE_LOCAL) >= 2:
        return STATE_LOCAL if claim_mode == "registered" else STATE_STRIPPED
    if STATE_MODAL in states:
        return STATE_MODAL if claim_mode == "registered" else STATE_STRIPPED
    if len(states) == 1 and states[0] == STATE_LOCAL:
        return STATE_REJECTED
    if all(state == STATE_VERIFIED for state in states):
        return STATE_VERIFIED if claim_mode == "registered" else STATE_REJECTED
    return STATE_REJECTED


def add_fused_states(df: pd.DataFrame, claim_mode: str, policy: str) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "m1_audio_event_state",
        "m1_audio_event_state_v1",
        "audio_event_long_state",
        "audio_event_short_state",
        "audio_id_state",
        "m1_sync_state",
        "m1_sync_state_v1",
        "video_whole_state",
        "video_whole_state_v1",
        "video_window_state",
        "video_window_state_v1",
        "audio_window_alarm_state",
    ]:
        if col not in out.columns:
            out[col] = ""

    if policy in {"v1", "v2", "v3"}:
        out["state_audio_event"] = out.apply(lambda row: fuse_audio_event_v1(row, claim_mode), axis=1)
    else:
        out["state_audio_event"] = out.apply(
            lambda row: fuse_branch(
                [
                    row.get("m1_audio_event_state"),
                    row.get("audio_event_long_state"),
                    row.get("audio_event_short_state"),
                ],
                claim_mode,
            ),
            axis=1,
        )
    out["state_audio_id"] = out["audio_id_state"].map(normalize_simple_state)
    sync_col = "m1_sync_state_v1" if policy in {"v1", "v2", "v3"} else "m1_sync_state"
    video_whole_col = "video_whole_state_v1" if policy in {"v1", "v2", "v3"} else "video_whole_state"
    video_window_col = "video_window_state_v1" if policy in {"v1", "v2", "v3"} else "video_window_state"
    out["state_sync"] = out[sync_col].map(normalize_simple_state)
    out["state_video_whole"] = out[video_whole_col].map(normalize_simple_state)
    out["state_video_window"] = out[video_window_col].map(normalize_simple_state)
    out["state_audio_window"] = out["audio_window_alarm_state"].map(normalize_simple_state)

    out["state_audio"] = out.apply(
        lambda row: fuse_branch([row["state_audio_event"], row["state_audio_id"]], claim_mode),
        axis=1,
    )
    if policy == "v2":
        out["state_video"] = out.apply(
            lambda row: fuse_branch([row["state_video_window"]], claim_mode),
            axis=1,
        )
        out["state_audio_sync"] = out["state_audio"]
        out["state_audio_video"] = out.apply(
            lambda row: fuse_branch([row["state_audio"], row["state_video"]], claim_mode),
            axis=1,
        )
        out["state_full_av"] = out["state_audio_video"]
    elif policy == "v3":
        out["state_audio"] = out.apply(
            lambda row: fuse_branch(
                [row["state_audio_event"], row["state_audio_id"], row["state_audio_window"]],
                claim_mode,
            ),
            axis=1,
        )
        out["state_video"] = out.apply(
            lambda row: fuse_branch([row["state_video_window"]], claim_mode),
            axis=1,
        )
        out["state_audio_sync"] = out["state_audio"]
        out["state_audio_video"] = out.apply(
            lambda row: fuse_branch([row["state_audio"], row["state_video"]], claim_mode),
            axis=1,
        )
        out["state_full_av"] = out["state_audio_video"]
    else:
        out["state_video"] = out.apply(
            lambda row: fuse_branch([row["state_video_whole"], row["state_video_window"]], claim_mode),
            axis=1,
        )
        out["state_audio_sync"] = out.apply(
            lambda row: fuse_branch([row["state_audio"], row["state_sync"]], claim_mode),
            axis=1,
        )
        out["state_audio_video"] = out.apply(
            lambda row: fuse_branch([row["state_audio"], row["state_video"]], claim_mode),
            axis=1,
        )
        out["state_full_av"] = out.apply(
            lambda row: fuse_branch([row["state_audio"], row["state_sync"], row["state_video"]], claim_mode),
            axis=1,
        )
    return out


def is_l1_benign(row: pd.Series) -> bool:
    return str(row.get("cond", "")).lower() == "l1" or str(row.get("attack_level", "")).upper() == "L1"


def is_attack_eval(row: pd.Series) -> bool:
    population = str(row.get("population", "")).lower()
    cond = str(row.get("cond", "")).lower()
    level = str(row.get("attack_level", "")).upper()
    if population == "orig" or cond == "clean":
        return False
    if is_l1_benign(row):
        return False
    # L2 neural processing is reported as robustness/usability unless the
    # manifest labels it as tamper. The security-facing epsilon table is for
    # true semantic tamper and L3/L4 regeneration laundering.
    return population in {"tamper", "attack", "different", "swap"} or level in {"L3", "L4"}


def summarize_states(df: pd.DataFrame, state_cols: Sequence[str]) -> pd.DataFrame:
    group_cols = [col for col in ["split", "population", "cond", "generator", "attack_level", "attack_family"] if col in df.columns]
    rows: List[Dict[str, object]] = []
    for state_col in state_cols:
        for keys, group in df.groupby(group_cols, dropna=False, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            base = dict(zip(group_cols, keys))
            counts = group[state_col].value_counts().to_dict()
            row = {
                "state_scope": state_col,
                **base,
                "n": int(len(group)),
                "verified": int(counts.get(STATE_VERIFIED, 0)),
                "content_unverified": int(counts.get(STATE_CONTENT, 0)),
                "modal_replacement": int(counts.get(STATE_MODAL, 0)),
                "localized_tamper": int(counts.get(STATE_LOCAL, 0)),
                "stripped_suspicious": int(counts.get(STATE_STRIPPED, 0)),
                "rejected": int(counts.get(STATE_REJECTED, 0)),
            }
            row["verified_rate"] = row["verified"] / row["n"] if row["n"] else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def epsilon_table(df: pd.DataFrame, state_cols: Sequence[str]) -> pd.DataFrame:
    attack = df[df.apply(is_attack_eval, axis=1)].copy()
    group_cols = [col for col in ["split", "population", "cond", "generator", "attack_level", "attack_family"] if col in attack.columns]
    rows: List[Dict[str, object]] = []
    if attack.empty:
        return pd.DataFrame(rows)
    for state_col in state_cols:
        for keys, group in attack.groupby(group_cols, dropna=False, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            false_verified = int((group[state_col] == STATE_VERIFIED).sum())
            n = int(len(group))
            lo, hi = cp95(false_verified, n)
            rows.append(
                {
                    "state_scope": state_col,
                    **dict(zip(group_cols, keys)),
                    "false_verified": false_verified,
                    "n": n,
                    "epsilon_sem_obl": false_verified / n if n else None,
                    "cp95_low": lo,
                    "cp95_high": hi,
                }
            )
    return pd.DataFrame(rows)


def benign_table(df: pd.DataFrame, state_cols: Sequence[str]) -> pd.DataFrame:
    benign = df[df.apply(is_l1_benign, axis=1)].copy()
    group_cols = [col for col in ["split", "population", "cond", "generator", "attack_level", "attack_family"] if col in benign.columns]
    rows: List[Dict[str, object]] = []
    if benign.empty:
        return pd.DataFrame(rows)
    for state_col in state_cols:
        for keys, group in benign.groupby(group_cols, dropna=False, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            verified = int((group[state_col] == STATE_VERIFIED).sum())
            n = int(len(group))
            lo, hi = cp95(verified, n)
            rows.append(
                {
                    "state_scope": state_col,
                    **dict(zip(group_cols, keys)),
                    "verified": verified,
                    "n": n,
                    "verified_rate": verified / n if n else None,
                    "cp95_low": lo,
                    "cp95_high": hi,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-fusion", type=Path, action="append", default=[])
    parser.add_argument("--audio-event-long", type=Path, action="append", default=[])
    parser.add_argument("--audio-event-short", type=Path, action="append", default=[])
    parser.add_argument("--audio-id", type=Path, action="append", default=[])
    parser.add_argument("--audio-window", type=Path, action="append", default=[])
    parser.add_argument("--video-whole", type=Path, action="append", default=[])
    parser.add_argument("--video-window", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--claim-mode",
        choices=["registered", "stripped"],
        default="registered",
        help="registered means a source claim/payload is recoverable in this Stage-0 run.",
    )
    parser.add_argument(
        "--policy",
        choices=["v0", "v1", "v2", "v3"],
        default="v0",
        help=(
            "v1 uses high-confidence sync, audio-event two-window failure, and video downgrade-only evidence. "
            "v2 additionally keeps sync and whole-video out of the final granting path. "
            "v3 adds a structured audio-window downgrade-only alarm."
        ),
    )
    parser.add_argument("--audio-window-long-run", type=int, default=7)
    parser.add_argument("--audio-window-short-run", type=int, default=2)
    parser.add_argument("--audio-window-max-failed", type=int, default=4)
    parser.add_argument("--audio-window-ham-threshold", type=int, default=80)
    parser.add_argument("--audio-window-q90-threshold", type=float, default=70.0)
    parser.add_argument("--audio-window-compact-failed-ratio", type=float, default=0.08)
    parser.add_argument("--audio-window-compact-ham-threshold", type=int, default=78)
    parser.add_argument("--audio-window-coherent-run", type=int, default=6)
    parser.add_argument("--audio-window-coherent-failed-ratio", type=float, default=0.25)
    parser.add_argument("--audio-window-coherent-ham-threshold", type=int, default=100)
    parser.add_argument("--audio-window-diffuse-failed-ratio", type=float, default=0.30)
    parser.add_argument("--audio-window-diffuse-recovery-ratio", type=float, default=0.55)
    parser.add_argument("--audio-window-diffuse-min-failed", type=int, default=12)
    parser.add_argument("--audio-window-global-failed-ratio", type=float, default=0.80)
    parser.add_argument("--audio-window-global-q90-threshold", type=float, default=100.0)
    parser.add_argument("--audio-window-global-ham-threshold", type=int, default=120)
    parser.add_argument("--audio-window-long-failed-ratio", type=float, default=0.60)
    args = parser.parse_args()

    frames = [
        load_m1_fusion(args.m1_fusion),
        load_audio_event(args.audio_event_long, "long"),
        load_audio_event(args.audio_event_short, "short"),
        load_audio_id(args.audio_id),
        load_audio_window_alarm(
            args.audio_window,
            args.audio_window_long_run,
            args.audio_window_short_run,
            args.audio_window_max_failed,
            args.audio_window_ham_threshold,
            args.audio_window_q90_threshold,
            args.audio_window_compact_failed_ratio,
            args.audio_window_compact_ham_threshold,
            args.audio_window_coherent_run,
            args.audio_window_coherent_failed_ratio,
            args.audio_window_coherent_ham_threshold,
            args.audio_window_diffuse_failed_ratio,
            args.audio_window_diffuse_recovery_ratio,
            args.audio_window_diffuse_min_failed,
            args.audio_window_global_failed_ratio,
            args.audio_window_global_q90_threshold,
            args.audio_window_global_ham_threshold,
            args.audio_window_long_failed_ratio,
        ),
        load_video_whole(args.video_whole),
        load_video_window(args.video_window),
    ]
    fused = add_fused_states(merge_evidence(frames), args.claim_mode, args.policy)

    state_cols = [
        "state_audio_event",
        "state_audio_id",
        "state_audio_window",
        "state_sync",
        "state_video_whole",
        "state_video_window",
        "state_audio",
        "state_video",
        "state_audio_sync",
        "state_audio_video",
        "state_full_av",
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fused.to_csv(args.out_dir / "m3_fused_rows.csv", index=False)
    summarize_states(fused, state_cols).to_csv(args.out_dir / "m3_state_summary.csv", index=False)
    epsilon_table(fused, state_cols).to_csv(args.out_dir / "m3_epsilon_sem_obl.csv", index=False)
    benign_table(fused, state_cols).to_csv(args.out_dir / "m3_benign_verified.csv", index=False)

    summary = {
        "rows": int(len(fused)),
        "claim_mode": args.claim_mode,
        "policy": args.policy,
        "state_columns": state_cols,
        "inputs": {
            "m1_fusion": [str(path) for path in args.m1_fusion],
            "audio_event_long": [str(path) for path in args.audio_event_long],
            "audio_event_short": [str(path) for path in args.audio_event_short],
            "audio_id": [str(path) for path in args.audio_id],
            "audio_window": [str(path) for path in args.audio_window],
            "video_whole": [str(path) for path in args.video_whole],
            "video_window": [str(path) for path in args.video_window],
        },
        "audio_window_rule": {
            "long_run": args.audio_window_long_run,
            "short_run": args.audio_window_short_run,
            "max_failed_windows": args.audio_window_max_failed,
            "ham_threshold": args.audio_window_ham_threshold,
            "q90_threshold": args.audio_window_q90_threshold,
            "compact_failed_ratio": args.audio_window_compact_failed_ratio,
            "compact_ham_threshold": args.audio_window_compact_ham_threshold,
            "coherent_run": args.audio_window_coherent_run,
            "coherent_failed_ratio": args.audio_window_coherent_failed_ratio,
            "coherent_ham_threshold": args.audio_window_coherent_ham_threshold,
            "diffuse_failed_ratio": args.audio_window_diffuse_failed_ratio,
            "diffuse_recovery_ratio": args.audio_window_diffuse_recovery_ratio,
            "diffuse_min_failed_windows": args.audio_window_diffuse_min_failed,
            "global_failed_ratio": args.audio_window_global_failed_ratio,
            "global_q90_threshold": args.audio_window_global_q90_threshold,
            "global_ham_threshold": args.audio_window_global_ham_threshold,
            "long_failed_ratio": args.audio_window_long_failed_ratio,
        },
        "threshold_provenance": (
            "M3 uses frozen upstream M1/M2 decisions. Thresholds must be calibrated "
            "only from orig + benign L1/L2 rows before L3/tamper evaluation."
        ),
        "scope_note": (
            "This estimates epsilon_sem^obl under generation-oblivious attacks. "
            "It is not an adaptive epsilon_sem^adapt estimate."
        ),
    }
    (args.out_dir / "m3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
