#!/usr/bin/env python3
"""Evaluate M1 audio-identity separability from cached ECAPA features.

This is the missing speaker-identity counterpart to evt_audio_v2. It is meant
for voice-conversion attacks such as FreeVC and kNN-VC: the transcript/content
may be preserved, but the speaker timbre should drift away from the enrolled
source identity.
"""

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


CHANNEL = "id_audio"
PATH_COLUMN = "id_audio_path"


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
                        help="optional planned rows for unavailable-feature accounting")
    parser.add_argument("--feature-failures", type=Path,
                        help="optional extraction failure CSV recorded for audit")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-bits", type=int, default=255)
    parser.add_argument("--master-key", default="rq1-m1-id-audio-stage0")
    parser.add_argument("--q-l1", type=float, default=0.99)
    parser.add_argument("--drop-missing-orig", action="store_true",
                        help="drop rows whose enroll_id has no extracted original feature")
    args = parser.parse_args()

    df = pd.read_csv(args.feature_manifest).fillna("")
    if PATH_COLUMN not in df:
        raise ValueError(f"feature manifest is missing {PATH_COLUMN}")
    feature_failures = args.feature_failures
    if (
        feature_failures
        and feature_failures.exists()
        and feature_failures.stat().st_mtime < args.feature_manifest.stat().st_mtime
    ):
        print(f"[m1-id-audio] ignoring stale feature-failure CSV: {feature_failures}", flush=True)
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
            f"[m1-id-audio] dropping {dropped_missing_orig} rows across "
            f"{len(missing)} enroll_id(s) with missing original features: {missing[:10]}",
            flush=True,
        )
        df = df[keep].copy()
        orig = df[df.population == "orig"].set_index("enroll_id", drop=False)

    features = {}
    for _, row in df.iterrows():
        path = str(row.get(PATH_COLUMN, ""))
        if path:
            features[str(row.sample_id)] = load_vector(path)
    if not features:
        raise ValueError("feature manifest has no available id_audio vectors")

    dim = len(next(iter(features.values())))
    qbank = QuantizerBank({CHANNEL: dim}, args.n_bits, kind="lsh",
                          master_key=args.master_key.encode())
    codes = {
        sample_id: qbank.encode(CHANNEL, vector)
        for sample_id, vector in features.items()
    }

    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        enrolled_sample = str(orig.loc[row.enroll_id].sample_id)
        current_sample = str(row.sample_id)
        if enrolled_sample not in codes or current_sample not in codes:
            continue
        hb = d_ham_bits(codes[enrolled_sample], codes[current_sample])
        rows.append({
            **row.to_dict(),
            "channel": CHANNEL,
            "reference_sample_id": enrolled_sample,
            "ham_bits": hb,
            "n": args.n_bits,
            "d_ham": hb / args.n_bits,
            "d_cos": d_cos(features[enrolled_sample], features[current_sample]),
        })

    orig_rows = list(orig.itertuples(index=False))
    for index, row in enumerate(orig_rows):
        partner = orig_rows[(index + 1) % len(orig_rows)]
        if partner.source_id == row.source_id and len(orig_rows) > 1:
            partner = orig_rows[(index + 2) % len(orig_rows)]
        current_sample = str(row.sample_id)
        partner_sample = str(partner.sample_id)
        if current_sample not in codes or partner_sample not in codes:
            continue
        hb = d_ham_bits(codes[current_sample], codes[partner_sample])
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
            "channel": CHANNEL,
            "reference_sample_id": row.sample_id,
            "ham_bits": hb,
            "n": args.n_bits,
            "d_ham": hb / args.n_bits,
            "d_cos": d_cos(features[current_sample], features[partner_sample]),
        })

    scores = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.out_dir / "m1_id_audio_scores.csv", index=False)

    l1_calib = scores[
        (scores.population == "same") & (scores.cond == "L1") & (scores.split == "calib")
    ]
    if len(l1_calib) == 0:
        raise ValueError("no calibration L1 controls for id_audio threshold")
    threshold = int(math.ceil(float(np.quantile(l1_calib.ham_bits, args.q_l1))))

    metric_rows = []
    for (split, population, cond, generator), group in scores.groupby(
        ["split", "population", "cond", "generator"], dropna=False,
    ):
        accepted = int((group.ham_bits <= threshold).sum())
        metric_rows.append({
            "split": split,
            "population": population,
            "cond": cond,
            "generator": generator,
            "n": len(group),
            "threshold_bits": threshold,
            "speaker_identity_accept_rate": accepted / len(group),
            "d_ham_median": float(group.d_ham.median()),
            "d_ham_p95": float(group.d_ham.quantile(0.95)),
            "d_cos_median": float(group.d_cos.median()),
            "d_cos_p95": float(group.d_cos.quantile(0.95)),
        })
    pd.DataFrame(metric_rows).to_csv(args.out_dir / "m1_id_audio_group_metrics.csv", index=False)

    sample_scores = scores[scores.population != "cross"].set_index("sample_id")
    sample_meta = df.drop_duplicates("sample_id").set_index("sample_id")
    decision_rows = []
    for sample_id, meta in sample_meta.iterrows():
        scored = sample_id in sample_scores.index
        bits = sample_scores.loc[sample_id].ham_bits if scored else None
        bits_available = scored and not pd.isna(bits)
        identity_ok = bool(bits_available and int(bits) <= threshold)
        decision_rows.append({
            "sample_id": sample_id,
            "enroll_id": meta.enroll_id,
            "source_id": meta.source_id,
            "split": meta.split,
            "population": meta.population,
            "cond": meta.cond,
            "generator": meta.generator,
            "attack_level": meta.get("attack_level", ""),
            "attack_family": meta.get("attack_family", ""),
            "attack_scope": meta.get("attack_scope", ""),
            "generator_role": meta.get("generator_role", ""),
            "id_audio_ham_bits": int(bits) if bits_available else None,
            "id_audio_available": bool(bits_available),
            "speaker_identity_consistent": bool(identity_ok),
            "id_audio_verified": bool(identity_ok),
            "decision_reason": "scored" if bits_available else "feature_unavailable",
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
        planned_missing_orig_enroll_ids = sorted(set(str(v) for v in planned_missing_orig.enroll_id))
        unavailable = planned[
            planned.enroll_id.isin(orig.index)
            & ~planned.sample_id.isin(df.sample_id)
        ]
        supported_unavailable_rows = int(len(unavailable))
        supported_unavailable_l3_rows = int((unavailable.attack_level == "L3").sum())
        unavailable_rows = []
        for _, row in unavailable.iterrows():
            unavailable_rows.append({
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
                "id_audio_ham_bits": None,
                "id_audio_available": False,
                "speaker_identity_consistent": False,
                "id_audio_verified": False,
                "decision_reason": "feature_unavailable",
                "feature_error": failure_errors.get(str(row.sample_id), ""),
            })
        if unavailable_rows:
            decisions = pd.concat([decisions, pd.DataFrame(unavailable_rows)], ignore_index=True)
    decisions.to_csv(args.out_dir / "m1_id_audio_decisions.csv", index=False)

    separability = []
    l1 = scores[
        (scores.population == "same") & (scores.cond == "L1") & (scores.split == "calib")
    ].d_ham.to_numpy()
    l3 = scores[(scores.attack_level == "L3") & (scores.attack_family == "audio_regeneration")]
    for generator in sorted(l3.generator.unique()):
        stressed = l3[l3.generator == generator].d_ham.to_numpy()
        e, eer_threshold = eer(l1, stressed)
        separability.append({
            "channel": CHANNEL,
            "generator": generator,
            "n_l1_calib": len(l1),
            "n_l3": len(stressed),
            "auc_l1_vs_l3": auc(l1, stressed),
            "eer_l1_vs_l3": e,
            "eer_threshold": eer_threshold,
            "dprime_l1_vs_l3": dprime(l1, stressed),
        })
    pd.DataFrame(separability).to_csv(args.out_dir / "m1_id_audio_separability.csv", index=False)

    summary = {
        "feature_rows": int(len(df)),
        "score_rows": int(len(scores)),
        "dropped_missing_orig_rows": dropped_missing_orig,
        "dropped_missing_orig_enroll_ids": dropped_missing_orig_enroll_ids,
        "planned_missing_orig_rows": planned_missing_orig_rows,
        "planned_missing_orig_enroll_ids": planned_missing_orig_enroll_ids,
        "supported_unavailable_rows": supported_unavailable_rows,
        "supported_unavailable_l3_rows": supported_unavailable_l3_rows,
        "threshold_bits": threshold,
        "l1_calib_id_audio_verified": cp_block(
            int(decisions[
                (decisions.population == "same")
                & (decisions.cond == "L1")
                & (decisions.split == "calib")
            ].id_audio_verified.sum()),
            int(len(decisions[
                (decisions.population == "same")
                & (decisions.cond == "L1")
                & (decisions.split == "calib")
            ])),
        ),
        "l3_by_generator": {},
        "current_scope": "speaker-identity commitment for voice-conversion attacks",
    }
    l3_decisions = decisions[
        (decisions.attack_level == "L3")
        & (decisions.attack_family == "audio_regeneration")
    ]
    for generator, group in l3_decisions.groupby("generator"):
        summary["l3_by_generator"][generator] = {
            "split": str(group.iloc[0].split),
            "generator_role": str(group.iloc[0].generator_role),
            "id_audio_available": cp_block(int(group.id_audio_available.sum()), int(len(group))),
            "id_audio_verified": cp_block(int(group.id_audio_verified.sum()), int(len(group))),
            "speaker_identity_consistent": cp_block(
                int(group.speaker_identity_consistent.sum()), int(len(group))
            ),
        }
    (args.out_dir / "m1_id_audio_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
