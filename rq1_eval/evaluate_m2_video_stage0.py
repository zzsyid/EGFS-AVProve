#!/usr/bin/env python3
"""Evaluate M2 video identity and visual-event separability from cached features."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from distances import d_cos, d_ham_bits
from epsilon_sem import clopper_pearson
from metrics_sep import auc, dprime, eer
from quantize import QuantizerBank


CHANNELS = ["id_video", "evt_video"]
PATH_COLUMNS = {channel: f"{channel}_path" for channel in CHANNELS}


def load_vector(path: str) -> np.ndarray:
    vector = np.load(path).astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError(f"zero feature vector: {path}")
    return vector / (norm + 1e-12)


def cp_block(events: int, n: int) -> Dict[str, object]:
    lo, hi = clopper_pearson(events, n)
    return {
        "events": int(events),
        "n": int(n),
        "point": (events / n if n else None),
        "clopper_pearson95": [lo, hi],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path,
                        help="optional planned M2 rows for fail-safe unavailable-feature accounting")
    parser.add_argument("--feature-failures", type=Path,
                        help="optional extraction failure CSV recorded for audit")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-bits", type=int, default=255)
    parser.add_argument("--master-key", default="rq1-m2-video-stage0")
    parser.add_argument("--q-l1", type=float, default=0.99)
    parser.add_argument("--drop-missing-orig", action="store_true",
                        help="drop rows whose enroll_id has no extracted original feature")
    args = parser.parse_args()

    df = pd.read_csv(args.feature_manifest).fillna("")
    feature_failures = args.feature_failures
    if (
        feature_failures
        and feature_failures.exists()
        and feature_failures.stat().st_mtime < args.feature_manifest.stat().st_mtime
    ):
        print(
            f"[m2-video] ignoring stale feature-failure CSV: {feature_failures}",
            flush=True,
        )
        feature_failures = None
    orig = df[df.population == "orig"].set_index("enroll_id", drop=False)
    if len(orig) == 0:
        raise ValueError("feature manifest has no enrolled originals")
    missing = sorted(set(df.enroll_id) - set(orig.index))
    dropped_missing_orig = 0
    dropped_missing_orig_enroll_ids: List[str] = []
    if missing:
        if not args.drop_missing_orig:
            raise ValueError(f"feature rows reference missing originals: {missing[:10]}")
        keep = df.enroll_id.isin(orig.index)
        dropped_missing_orig = int((~keep).sum())
        dropped_missing_orig_enroll_ids = missing
        print(
            f"[m2-video] dropping {dropped_missing_orig} rows across "
            f"{len(missing)} enroll_id(s) with missing original features: {missing[:10]}",
            flush=True,
        )
        df = df[keep].copy()
        orig = df[df.population == "orig"].set_index("enroll_id", drop=False)

    features = {}
    for _, row in df.iterrows():
        channels = {}
        for channel in CHANNELS:
            path = str(row.get(PATH_COLUMNS[channel], ""))
            if path:
                channels[channel] = load_vector(path)
        features[str(row.sample_id)] = channels
    dims = {}
    for channel in CHANNELS:
        vectors = [
            channels[channel] for channels in features.values() if channel in channels
        ]
        if not vectors:
            raise ValueError(f"{channel}: feature manifest has no available vectors")
        dims[channel] = len(vectors[0])
    qbank = QuantizerBank(dims, args.n_bits, kind="lsh", master_key=args.master_key.encode())
    codes = {
        sample_id: {channel: qbank.encode(channel, vector) for channel, vector in channels.items()}
        for sample_id, channels in features.items()
    }

    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        enrolled_sample = str(orig.loc[row.enroll_id].sample_id)
        current_sample = str(row.sample_id)
        for channel in CHANNELS:
            if channel not in codes[enrolled_sample] or channel not in codes[current_sample]:
                continue
            hb = d_ham_bits(codes[enrolled_sample][channel], codes[current_sample][channel])
            rows.append({
                **row.to_dict(),
                "channel": channel,
                "reference_sample_id": enrolled_sample,
                "ham_bits": hb,
                "n": args.n_bits,
                "d_ham": hb / args.n_bits,
                "d_cos": d_cos(features[enrolled_sample][channel], features[current_sample][channel]),
            })

    orig_rows = list(orig.itertuples(index=False))
    for index, row in enumerate(orig_rows):
        partner = orig_rows[(index + 1) % len(orig_rows)]
        if partner.source_id == row.source_id and len(orig_rows) > 1:
            partner = orig_rows[(index + 2) % len(orig_rows)]
        current_sample = str(row.sample_id)
        partner_sample = str(partner.sample_id)
        for channel in CHANNELS:
            if channel not in codes[current_sample] or channel not in codes[partner_sample]:
                continue
            hb = d_ham_bits(codes[current_sample][channel], codes[partner_sample][channel])
            rows.append({
                "sample_id": f"cross::{row.sample_id}::{partner.sample_id}",
                "enroll_id": row.enroll_id,
                "source_id": row.source_id,
                "population": "cross",
                "cond": "cross_source",
                "generator": "virtual_cross_pair",
                "split": row.split,
                "attack_level": "cross",
                "attack_family": "cross_source",
                "attack_scope": "full",
                "generator_role": "shared",
                "channel": channel,
                "reference_sample_id": row.sample_id,
                "ham_bits": hb,
                "n": args.n_bits,
                "d_ham": hb / args.n_bits,
                "d_cos": d_cos(features[current_sample][channel], features[partner_sample][channel]),
            })

    scores = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.out_dir / "m2_video_scores.csv", index=False)

    thresholds = {}
    metric_rows = []
    decision_rows = []
    for channel in CHANNELS:
        cdf = scores[scores.channel == channel]
        l1_calib = cdf[
            (cdf.population == "same") & (cdf.cond == "L1") & (cdf.split == "calib")
        ]
        if len(l1_calib) == 0:
            raise ValueError(f"{channel}: no calibration L1 controls")
        threshold = int(math.ceil(float(np.quantile(l1_calib.ham_bits, args.q_l1))))
        thresholds[channel] = threshold
        for (split, population, cond, generator), group in cdf.groupby(
            ["split", "population", "cond", "generator"], dropna=False,
        ):
            accepted = int((group.ham_bits <= threshold).sum())
            metric_rows.append({
                "channel": channel,
                "split": split,
                "population": population,
                "cond": cond,
                "generator": generator,
                "n": len(group),
                "threshold_bits": threshold,
                "accept_rate": accepted / len(group),
                "d_ham_median": float(group.d_ham.median()),
                "d_ham_p95": float(group.d_ham.quantile(0.95)),
                "d_cos_median": float(group.d_cos.median()),
                "d_cos_p95": float(group.d_cos.quantile(0.95)),
            })

    sample_scores = scores[scores.population != "cross"].pivot(
        index="sample_id", columns="channel", values="ham_bits",
    )
    sample_meta = df.drop_duplicates("sample_id").set_index("sample_id")
    for sample_id, meta in sample_meta.iterrows():
        values = sample_scores.loc[sample_id] if sample_id in sample_scores.index else {}
        id_bits = values.get("id_video") if hasattr(values, "get") else None
        evt_bits = values.get("evt_video") if hasattr(values, "get") else None
        id_available = id_bits is not None and not pd.isna(id_bits)
        evt_available = evt_bits is not None and not pd.isna(evt_bits)
        identity_ok = bool(id_available and int(id_bits) <= thresholds["id_video"])
        event_ok = bool(evt_available and int(evt_bits) <= thresholds["evt_video"])
        if id_available and evt_available:
            decision_reason = "scored"
        elif id_available or evt_available:
            decision_reason = "partial_feature_unavailable"
        else:
            decision_reason = "feature_unavailable"
        decision_rows.append({
            "sample_id": sample_id,
            "enroll_id": meta.enroll_id,
            "source_id": meta.source_id,
            "split": meta.split,
            "population": meta.population,
            "cond": meta.cond,
            "generator": meta.generator,
            "attack_level": meta.attack_level,
            "attack_family": meta.attack_family,
            "attack_scope": meta.attack_scope,
            "generator_role": meta.generator_role,
            "id_video_ham_bits": int(id_bits) if id_available else None,
            "evt_video_ham_bits": int(evt_bits) if evt_available else None,
            "id_video_available": bool(id_available),
            "evt_video_available": bool(evt_available),
            "identity_consistent": bool(identity_ok),
            "visual_event_consistent": bool(event_ok) if evt_available else None,
            "video_verified": bool(identity_ok and event_ok),
            "feature_available": bool(id_available and evt_available),
            "decision_reason": decision_reason,
            "feature_error": "; ".join(
                error for error in (
                    str(meta.get("id_video_error", "")),
                    str(meta.get("evt_video_error", "")),
                ) if error
            ),
        })
    decisions = pd.DataFrame(decision_rows)

    planned_missing_orig_rows = 0
    planned_missing_orig_enroll_ids: List[str] = []
    supported_unavailable_rows = 0
    supported_unavailable_l3_rows = 0
    if args.pilot_manifest:
        planned = pd.read_csv(args.pilot_manifest).fillna("")
        failure_errors = {}
        if feature_failures and feature_failures.exists():
            failures = pd.read_csv(feature_failures).fillna("")
            failure_errors = {
                str(row.sample_id): str(row.error) for _, row in failures.iterrows()
            }
        planned_missing_orig = planned[~planned.enroll_id.isin(orig.index)]
        planned_missing_orig_rows = int(len(planned_missing_orig))
        planned_missing_orig_enroll_ids = sorted(
            set(str(value) for value in planned_missing_orig.enroll_id)
        )
        unavailable = planned[
            planned.enroll_id.isin(orig.index)
            & ~planned.sample_id.isin(df.sample_id)
        ]
        supported_unavailable_rows = int(len(unavailable))
        supported_unavailable_l3_rows = int(
            ((unavailable.attack_level == "L3") & (unavailable.attack_family == "video_regeneration")).sum()
        )
        if len(unavailable):
            unavailable_decisions = []
            for _, row in unavailable.iterrows():
                unavailable_decisions.append({
                    "sample_id": row.sample_id,
                    "enroll_id": row.enroll_id,
                    "source_id": row.source_id,
                    "split": row.split,
                    "population": row.population,
                    "cond": row.cond,
                    "generator": row.generator,
                    "attack_level": row.attack_level,
                    "attack_family": row.attack_family,
                    "attack_scope": row.attack_scope,
                    "generator_role": row.generator_role,
                    "id_video_ham_bits": None,
                    "evt_video_ham_bits": None,
                    "id_video_available": False,
                    "evt_video_available": False,
                    "identity_consistent": False,
                    "visual_event_consistent": None,
                    "video_verified": False,
                    "feature_available": False,
                    "decision_reason": "feature_unavailable",
                    "feature_error": failure_errors.get(str(row.sample_id), ""),
                })
            decisions = pd.concat(
                [decisions, pd.DataFrame(unavailable_decisions)],
                ignore_index=True,
            )
    decisions.to_csv(args.out_dir / "m2_video_decisions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(args.out_dir / "m2_video_group_metrics.csv", index=False)

    separability = []
    for channel in CHANNELS:
        cdf = scores[scores.channel == channel]
        l1 = cdf[(cdf.population == "same") & (cdf.cond == "L1") & (cdf.split == "calib")].d_ham.to_numpy()
        for generator in sorted(cdf[(cdf.attack_level == "L3") & (cdf.attack_family == "video_regeneration")].generator.unique()):
            stressed = cdf[(cdf.generator == generator) & (cdf.attack_level == "L3")].d_ham.to_numpy()
            e, threshold = eer(l1, stressed)
            separability.append({
                "channel": channel,
                "generator": generator,
                "n_l1_calib": len(l1),
                "n_l3": len(stressed),
                "auc_l1_vs_l3": auc(l1, stressed),
                "eer_l1_vs_l3": e,
                "eer_threshold": threshold,
                "dprime_l1_vs_l3": dprime(l1, stressed),
            })
    pd.DataFrame(separability).to_csv(args.out_dir / "m2_video_separability.csv", index=False)

    summary = {
        "feature_rows": int(len(df)),
        "score_rows": int(len(scores)),
        "dropped_missing_orig_rows": dropped_missing_orig,
        "dropped_missing_orig_enroll_ids": dropped_missing_orig_enroll_ids,
        "planned_missing_orig_rows": planned_missing_orig_rows,
        "planned_missing_orig_enroll_ids": planned_missing_orig_enroll_ids,
        "supported_unavailable_rows": supported_unavailable_rows,
        "supported_unavailable_l3_rows": supported_unavailable_l3_rows,
        "partial_feature_rows": int((decisions.decision_reason == "partial_feature_unavailable").sum()),
        "partial_feature_l3_rows": int(
            (
                (decisions.decision_reason == "partial_feature_unavailable")
                & (decisions.attack_level == "L3")
                & (decisions.attack_family == "video_regeneration")
            ).sum()
        ),
        "feature_failures_csv": str(feature_failures) if feature_failures else None,
        "thresholds_bits": thresholds,
        "l1_calib_video_verified": cp_block(
            int(decisions[(decisions.population == "same") & (decisions.cond == "L1") & (decisions.split == "calib")].video_verified.sum()),
            int(len(decisions[(decisions.population == "same") & (decisions.cond == "L1") & (decisions.split == "calib")])),
        ),
        "l3_by_generator": {},
        "decision_by_condition": {},
        "current_scope": "talking-head redraw and cross-source reenactment drift pilot",
        "required_next_attacks": ["face_replacement", "local_video_splice", "frame_level_redraw_or_purification"],
    }
    l3_video = decisions[
        (decisions.attack_level == "L3")
        & (decisions.attack_family == "video_regeneration")
    ]
    for generator, group in l3_video.groupby("generator"):
        event_group = group[group.visual_event_consistent.notna()]
        summary["l3_by_generator"][generator] = {
            "split": str(group.iloc[0].split),
            "population": str(group.iloc[0].population),
            "generator_role": str(group.iloc[0].generator_role),
            "feature_available": cp_block(int(group.feature_available.sum()), int(len(group))),
            "id_video_available": cp_block(int(group.id_video_available.sum()), int(len(group))),
            "evt_video_available": cp_block(int(group.evt_video_available.sum()), int(len(group))),
            "video_verified": cp_block(int(group.video_verified.sum()), int(len(group))),
            "identity_consistent": cp_block(int(group.identity_consistent.sum()), int(len(group))),
            "visual_event_consistent": cp_block(
                int(event_group.visual_event_consistent.sum()), int(len(event_group))
            ),
        }
    conditions = decisions[decisions.population != "orig"]
    for (cond, generator), group in conditions.groupby(["cond", "generator"]):
        key = f"{cond}::{generator}"
        summary["decision_by_condition"][key] = {
            "n": int(len(group)),
            "feature_available": cp_block(int(group.feature_available.sum()), int(len(group))),
            "video_verified": cp_block(int(group.video_verified.sum()), int(len(group))),
        }
    (args.out_dir / "m2_video_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
