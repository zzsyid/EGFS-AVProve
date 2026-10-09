#!/usr/bin/env python3
"""Verify M6 media-realized weak-adaptive rows through signed M4 claims.

Unlike the frozen M4 replay verifier, M6 sample_ids are new generated media.
The claim being tested is therefore the enrolled source claim carried by each
M6 row (`enroll_id`, `source_id`). We verify the signed registry record for the
enrollment, enforce source-scope binding, then emit the freshly recomputed M3
state for the realized media.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from epsilon_sem import clopper_pearson
from registry_signatures import canonical, load_public, read_jsonl, unsigned

STATE_VERIFIED = "source-and-content verified"
STATE_MODAL = "tampered/modal-replacement"
LEGACY_STATE_MODAL = "modal-replacement"
STATE_SIGNATURE_FAILED = "signature-failed"
STATE_MISSING_REGISTRY = "missing-registry"
STATE_REJECTED = "rejected"
LEGACY_STATE_REJECTED = "unknown"

GROUP_COLS = [
    "m6_attack_type",
    "m6_candidate_policy",
    "generator",
    "cond",
    "split",
    "m6_budget_bits",
]


def has_text(value: object) -> bool:
    text = str(value).strip().lower()
    return text not in {"", "nan", "none"}


def verify_signature(public: Any, record: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        payload_record = unsigned(record)
        payload = canonical(payload_record)
        if hashlib.sha256(payload).hexdigest() != record.get("record_sha256"):
            return False, "record_sha256_mismatch"
        signature = base64.b64decode(record["signature_ed25519_b64"], validate=True)
        public.verify(signature, payload)
        return True, "signature_ok"
    except Exception as exc:
        return False, f"signature_exception:{type(exc).__name__}"


def load_registry(path: Path, public_key: Path) -> Dict[str, Dict[str, object]]:
    public = load_public(public_key)
    out: Dict[str, Dict[str, object]] = {}
    for envelope in read_jsonl(path):
        ok, reason = verify_signature(public, envelope)
        manifest = dict(envelope.get("manifest", {}))
        manifest["_signature_ok"] = ok
        manifest["_signature_reason"] = reason
        enroll_id = str(manifest.get("enroll_id", ""))
        if enroll_id:
            out[enroll_id] = manifest
    return out


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cp_block(k: int, n: int) -> Dict[str, object]:
    lo, hi = clopper_pearson(k, n)
    return {
        "events": int(k),
        "n": int(n),
        "point": k / n if n else None,
        "cp95_low": lo,
        "cp95_high": hi,
    }


def relevant_modality_unknown(row: pd.Series) -> Tuple[bool, str]:
    """Return whether M6 should refuse verified due to missing relevant modality evidence."""
    attack_type = str(row.get("m6_attack_type", "")).lower()
    generator = str(row.get("generator", "")).lower()
    family = str(row.get("attack_family", "")).lower()
    video_relevant = (
        "video" in attack_type
        or "face" in attack_type
        or generator in {
            "liveportrait",
            "aniportrait",
            "lavdf_fake_av",
            "ffmpeg_video_splice",
            "ffmpeg_full_video_replacement",
        }
        or family in {"video_regeneration", "av_recomposition", "local_video_edit", "face_replacement"}
    )
    audio_relevant = (
        "audio" in attack_type
        or generator in {"knn-vc", "freevc", "xtts", "ffmpeg_splice"}
        or family in {"audio_regeneration", "local_edit", "local_audio_edit"}
    )
    if video_relevant and str(row.get("state_video", "")).strip().lower() in {"", "unknown", "nan", "none"}:
        return True, "m6_relevant_video_evidence_unknown"
    if audio_relevant and str(row.get("state_audio", "")).strip().lower() in {"", "unknown", "nan", "none"}:
        return True, "m6_relevant_audio_evidence_unknown"
    return False, ""


def state_counts(group: pd.DataFrame) -> Dict[str, int]:
    counts = group["final_state"].astype(str).value_counts().to_dict()
    return {
        "source_and_content_verified": int(counts.get(STATE_VERIFIED, 0)),
        "localized_tamper": int(counts.get("localized-tamper", 0)),
        "content_unverified": int(counts.get("content-unverified", 0)),
        "modal_replacement": int(counts.get(STATE_MODAL, 0)),
        "rejected": int(counts.get(STATE_REJECTED, 0)),
        "signature_failed": int(counts.get(STATE_SIGNATURE_FAILED, 0)),
        "missing_registry": int(counts.get(STATE_MISSING_REGISTRY, 0)),
    }


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        return []
    df = pd.DataFrame(rows).fillna("")
    for col in GROUP_COLS:
        if col not in df.columns:
            df[col] = ""
    out: List[Dict[str, object]] = []
    for keys, group in df.groupby(GROUP_COLS, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = int(len(group))
        false_verified = int((group["final_state"] == STATE_VERIFIED).sum())
        block = cp_block(false_verified, n)
        out.append({
            **dict(zip(GROUP_COLS, keys)),
            "n": n,
            "false_verified": false_verified,
            "epsilon_sem_weak_adapt_media": block["point"],
            "cp95_low": block["cp95_low"],
            "cp95_high": block["cp95_high"],
            **state_counts(group),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--signed-registry", type=Path, required=True)
    ap.add_argument("--public-key", type=Path, required=True)
    ap.add_argument("--m3-fused", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--state-col", default="state_full_av")
    ap.add_argument(
        "--m6-only",
        action="store_true",
        help="verify only rows carrying M6 metadata, ignoring baseline calibration rows",
    )
    ap.add_argument(
        "--require-m6-relevant-modality",
        action="store_true",
        help="for M6 rows, refuse source-and-content verified if the attacked modality has unknown M3 evidence",
    )
    args = ap.parse_args()

    registry = load_registry(args.signed_registry, args.public_key)
    m3 = pd.read_csv(args.m3_fused).fillna("")
    if args.state_col not in m3.columns:
        raise ValueError(f"M3 fused CSV missing {args.state_col}")

    rows: List[Dict[str, object]] = []
    for _, row in m3.iterrows():
        if args.m6_only and not has_text(row.get("m6_attack_type", "")):
            continue
        enroll_id = str(row.get("enroll_id", ""))
        record = registry.get(enroll_id)
        m3_state = str(row.get(args.state_col, STATE_REJECTED) or STATE_REJECTED)
        if m3_state == LEGACY_STATE_MODAL:
            m3_state = STATE_MODAL
        elif m3_state == LEGACY_STATE_REJECTED:
            m3_state = STATE_REJECTED
        if record is None:
            final_state = STATE_MISSING_REGISTRY
            signature_ok = False
            source_scope_ok = False
            reason = "missing_signed_registry_record_for_enroll_id"
            registry_source_id = ""
        else:
            signature_ok = bool(record.get("_signature_ok", False))
            registry_source_id = str(record.get("source_id", ""))
            source_scope_ok = registry_source_id == str(row.get("source_id", ""))
            if not signature_ok:
                final_state = STATE_SIGNATURE_FAILED
                reason = str(record.get("_signature_reason", "signature_failed"))
            elif not source_scope_ok:
                final_state = STATE_MODAL
                reason = "signed_source_scope_mismatch"
            else:
                final_state = m3_state
                reason = "signature_source_scope_and_recomputed_m3_media_evidence_passed_to_fusion"
                if args.require_m6_relevant_modality and final_state == STATE_VERIFIED:
                    missing_modality, modality_reason = relevant_modality_unknown(row)
                    if missing_modality:
                        final_state = STATE_REJECTED
                        reason = modality_reason
        out = {
            "sample_id": str(row.get("sample_id", "")),
            "enroll_id": enroll_id,
            "source_id": str(row.get("source_id", "")),
            "registry_source_id": registry_source_id,
            "split": str(row.get("split", "")),
            "population": str(row.get("population", "")),
            "cond": str(row.get("cond", "")),
            "generator": str(row.get("generator", "")),
            "attack_level": str(row.get("attack_level", "")),
            "attack_family": str(row.get("attack_family", "")),
            "m6_attack_type": str(row.get("m6_attack_type", "")),
            "m6_candidate_policy": str(row.get("m6_candidate_policy", "")),
            "m6_candidate_source": str(row.get("m6_candidate_source", "")),
            "m6_target_sample_id": str(row.get("m6_target_sample_id", "")),
            "m6_donor_sample_id": str(row.get("m6_donor_sample_id", "")),
            "m6_target_source_id": str(row.get("m6_target_source_id", "")),
            "m6_donor_source_id": str(row.get("m6_donor_source_id", "")),
            "m6_budget_bits": str(row.get("m6_budget_bits", "")),
            "m6_bits_needed_to_pass": str(row.get("m6_bits_needed_to_pass", "")),
            "signature_ok": signature_ok,
            "source_scope_ok": source_scope_ok,
            "m3_state": m3_state,
            "final_state": final_state,
            "m3_parity_when_signed_scope_ok": bool(signature_ok and source_scope_ok and final_state == m3_state),
            "decision_reason": reason,
        }
        rows.append(out)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "m6_signed_m4_media_realized_rows.csv"
    summary_path = args.out_dir / "m6_signed_m4_media_realized_summary.csv"
    json_path = args.out_dir / "m6_signed_m4_media_realized_summary.json"
    write_csv(rows_path, rows)
    summary_rows = summarize(rows)
    write_csv(summary_path, summary_rows)
    eval_rows = [row for row in rows if str(row.get("split", "")) == "eval"]
    false_verified = sum(1 for row in eval_rows if row.get("final_state") == STATE_VERIFIED)
    report = {
        "scope": "M6 media-realized weak-adaptive verification through signed registry claim and recomputed M3 media evidence.",
        "m3_fused": str(args.m3_fused),
        "signed_registry": str(args.signed_registry),
        "rows": len(rows),
        "m6_only": bool(args.m6_only),
        "require_m6_relevant_modality": bool(args.require_m6_relevant_modality),
        "eval_false_verified": cp_block(false_verified, len(eval_rows)),
        "signature_failures": sum(1 for row in rows if not bool(row.get("signature_ok"))),
        "source_scope_failures": sum(1 for row in rows if not bool(row.get("source_scope_ok"))),
        "outputs": {
            "rows_csv": str(rows_path),
            "summary_csv": str(summary_path),
            "summary_json": str(json_path),
        },
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
