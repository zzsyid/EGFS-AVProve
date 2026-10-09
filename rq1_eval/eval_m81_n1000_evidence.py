#!/usr/bin/env python3
"""Evaluate a frozen M8.1 checkpoint on the closed N1000 evidence subset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from train_m81_windowed_fs_avprove_net import (
    FEAT_DIM,
    PREDICATES,
    STATES,
    STATE_TO_ID,
    TOKEN_TYPES,
    REJECTED,
    VERIFIED,
    HierarchicalEvidenceGraph,
    group_by_sample,
    index_by_sample,
    make_example,
    materialize_dataset,
    read_csv,
    run_eval,
    write_csv,
    write_json,
)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def distribution(values: np.ndarray) -> Dict[str, int]:
    return dict(Counter(STATES[int(value)] for value in values.tolist()))


def rate(events: int, n: int) -> float:
    return float(events / n) if n else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("rq1_eval/artifacts_m7_m6_e2e_scale1000/incremental")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--m3-fused", type=Path, default=root / "m3_fusion/m3_fused_rows_m6_m7_enriched.csv")
    parser.add_argument("--video-window-scores", type=Path, default=root / "video_window_evaluation/m2_video_window_scores.csv")
    parser.add_argument("--audio-window-scores", type=Path, default=root / "audio_window_span05_hop025/segmented_evt_bch_scores.csv")
    parser.add_argument("--claim-rows", type=Path, default=root / "claim_recovery/m7_short_pointer_claim_rows.csv")
    parser.add_argument("--signed-m4-rows", type=Path, default=root / "signed_m7_m4_verification/m6_signed_m4_media_realized_rows.csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-video-windows", type=int, default=48)
    parser.add_argument("--max-audio-failed-windows", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    architecture = checkpoint["architecture"]
    if architecture.get("encoder") != "hierarchical_evidence_graph":
        raise ValueError(
            "checkpoint does not use the released hierarchical evidence graph"
        )
    model = HierarchicalEvidenceGraph(
        FEAT_DIM,
        int(architecture["hidden_dim"]),
        int(architecture["num_heads"]),
        int(architecture["layers"]),
        float(architecture["dropout"]),
        int(architecture["max_tokens"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    m3_by_sample = index_by_sample(read_csv(args.m3_fused))
    claim_by_sample = index_by_sample(read_csv(args.claim_rows))
    audio_window_by_sample = index_by_sample(read_csv(args.audio_window_scores))
    video_window_by_sample = group_by_sample(read_csv(args.video_window_scores))
    signed_rows = read_csv(args.signed_m4_rows)

    examples = []
    retained_signed_rows: List[Dict[str, str]] = []
    missing_m3: List[str] = []
    for signed in signed_rows:
        sample_id = str(signed.get("sample_id", ""))
        m3 = m3_by_sample.get(sample_id)
        if not m3:
            missing_m3.append(sample_id)
            continue
        claim = claim_by_sample.get(sample_id, {})
        examples.append(
            make_example(
                m3,
                "signed_n1000",
                str(signed.get("final_state", m3.get("state_full_av", REJECTED))),
                claim,
                signed,
                {},
                {},
                audio_window_by_sample.get(sample_id, {}),
                video_window_by_sample.get(sample_id, []),
                args.max_video_windows,
                args.max_audio_failed_windows,
            )
        )
        retained_signed_rows.append(signed)

    data = materialize_dataset(examples, int(architecture["max_tokens"]), train_ratio=0.8)
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32).reshape(1, 1, FEAT_DIM)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32).reshape(1, 1, FEAT_DIM)
    data["tokens"] = ((data["tokens"] - mean) / std).astype(np.float32)
    data["normalization_mean"] = mean.reshape(-1)
    data["normalization_std"] = std.reshape(-1)

    indices = np.arange(len(examples), dtype=np.int64)
    metrics, outputs = run_eval(model, data, indices, device, args.batch_size)
    labels = data["state_labels"]
    raw_pred = outputs["logits"].argmax(axis=1)
    masked_pred = outputs["masked_logits"].argmax(axis=1)
    unsafe = labels != STATE_TO_ID[VERIFIED]

    rows: List[Dict[str, Any]] = []
    by_attack: Dict[str, List[int]] = defaultdict(list)
    for i, signed in enumerate(retained_signed_rows):
        attack = str(signed.get("generator", signed.get("cond", "unknown")))
        by_attack[attack].append(i)
        rows.append({
            "sample_id": signed.get("sample_id", ""),
            "generator": attack,
            "condition": signed.get("cond", ""),
            "m4_state": STATES[int(labels[i])],
            "m81_raw_state": STATES[int(raw_pred[i])],
            "m81_masked_state": STATES[int(masked_pred[i])],
            "raw_exact_match": bool(raw_pred[i] == labels[i]),
            "masked_exact_match": bool(masked_pred[i] == labels[i]),
            "raw_false_verified": bool(raw_pred[i] == STATE_TO_ID[VERIFIED] and unsafe[i]),
            "masked_false_verified": bool(masked_pred[i] == STATE_TO_ID[VERIFIED] and unsafe[i]),
            "verified_allowed_by_hard_mask": bool(data["state_masks"][i, STATE_TO_ID[VERIFIED]]),
        })

    by_attack_summary: Dict[str, Any] = {}
    for attack, members in sorted(by_attack.items()):
        idx = np.asarray(members, dtype=np.int64)
        raw_match = int((raw_pred[idx] == labels[idx]).sum())
        masked_match = int((masked_pred[idx] == labels[idx]).sum())
        by_attack_summary[attack] = {
            "n": int(idx.size),
            "m4_distribution": distribution(labels[idx]),
            "m81_raw_distribution": distribution(raw_pred[idx]),
            "m81_masked_distribution": distribution(masked_pred[idx]),
            "raw_exact_match": raw_match,
            "raw_exact_match_rate": rate(raw_match, int(idx.size)),
            "masked_exact_match": masked_match,
            "masked_exact_match_rate": rate(masked_match, int(idx.size)),
            "raw_false_verified": int(((raw_pred[idx] == STATE_TO_ID[VERIFIED]) & unsafe[idx]).sum()),
            "masked_false_verified": int(((masked_pred[idx] == STATE_TO_ID[VERIFIED]) & unsafe[idx]).sum()),
        }

    n = int(labels.size)
    raw_match = int((raw_pred == labels).sum())
    masked_match = int((masked_pred == labels).sum())
    summary = {
        "scope": "Frozen M8.1 inference over the closed signed N1000 evidence subset; no media generation or feature extraction rerun.",
        "checkpoint": str(args.checkpoint),
        "closed_signed_rows": len(signed_rows),
        "evaluated_rows": n,
        "missing_m3_rows": missing_m3,
        "m4_distribution": distribution(labels),
        "m81_raw_distribution": distribution(raw_pred),
        "m81_masked_distribution": distribution(masked_pred),
        "raw_exact_match": raw_match,
        "raw_exact_match_rate": rate(raw_match, n),
        "masked_exact_match": masked_match,
        "masked_exact_match_rate": rate(masked_match, n),
        "raw_false_verified": int(((raw_pred == STATE_TO_ID[VERIFIED]) & unsafe).sum()),
        "masked_false_verified": int(((masked_pred == STATE_TO_ID[VERIFIED]) & unsafe).sum()),
        "raw_safe_routing_rate": rate(int(((raw_pred != STATE_TO_ID[VERIFIED]) & unsafe).sum()), int(unsafe.sum())),
        "masked_safe_routing_rate": rate(int(((masked_pred != STATE_TO_ID[VERIFIED]) & unsafe).sum()), int(unsafe.sum())),
        "hard_mask_disallowed_verified": int((~data["state_masks"][:, STATE_TO_ID[VERIFIED]]).sum()),
        "model_metrics": metrics,
        "by_attack": by_attack_summary,
        "state_definitions": STATES,
        "predicate_definitions": PREDICATES,
        "token_types": TOKEN_TYPES,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "m81_n1000_predictions.csv", rows)
    write_json(args.out_dir / "m81_n1000_summary.json", summary)
    (args.out_dir / "M81_N1000_SUMMARY.md").write_text(
        "# M8.1 N1000 Evidence Evaluation\n\n\x60\x60\x60json\n"
        + json.dumps(summary, indent=2, sort_keys=True)
        + "\n\x60\x60\x60\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
