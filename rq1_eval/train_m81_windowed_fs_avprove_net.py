#!/usr/bin/env python3
"""Train M8.1 Windowed FS-AVProve-Net.

M8.1 upgrades the previous summary-stat MLP prototype into a graph-structured,
window-aware evidence verifier:

  claim/carrier/semantic/window/availability evidence graph
    -> token-level dependency routing
    -> hierarchical typed-slot graph reasoning
    -> predicate-factorized provenance estimates
    -> positive-support constrained state decoder
    -> hard safety-masked final state

The model never trains or replaces the cryptographic registry or the frozen
M3/M4 semantic verifier. Learned logits are masked by signed/source-scope and
semantic tamper gates before a verified state is allowed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn


POLICY = "tdsc-m8.1-evidence-graph-fs-avprove-net-v0.3"
VERIFIED = "source-and-content verified"
MODAL_REPLACEMENT = "tampered/modal-replacement"
REJECTED = "rejected"
LEGACY_REJECTED = "unknown"
STATES = [
    VERIFIED,
    "source-only verified",
    "localized-tamper",
    MODAL_REPLACEMENT,
    "content-unverified",
    REJECTED,
]
STATE_TO_ID = {name: i for i, name in enumerate(STATES)}

TOKEN_TYPES = {
    "claim": 0,
    "audio_carrier": 1,
    "video_carrier": 2,
    "audio_window": 3,
    "video_window": 4,
    "semantic_summary": 5,
    "sync": 6,
    "availability": 7,
}
FEAT_DIM = 24

PREDICATES = [
    "claim_authentic",
    "source_scope_valid",
    "audio_carrier_reliable",
    "video_carrier_reliable",
    "audio_content_consistent",
    "video_content_consistent",
    "sync_consistent",
    "local_tamper",
    "missing_evidence",
    "modal_conflict",
    "sufficient_video_support",
]
PRED_TO_ID = {name: i for i, name in enumerate(PREDICATES)}


@dataclass
class Example:
    sample_id: str
    source: str
    tokens: List[List[float]]
    token_types: List[int]
    loc_labels: List[float]
    state_label: int
    audio_label: float
    video_label: float
    signature_ok: float
    source_scope_ok: float
    claim_recovered: float
    semantic_tamper: float
    predicate_labels: List[float]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_many_csv(patterns: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).exists():
            matches = [pattern]
        for match in matches:
            rows.extend(read_csv(Path(match)))
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def parse_bool(value: Any) -> float:
    return 1.0 if str(value).strip().lower() in {"1", "true", "yes", "y"} else 0.0


def stable_bucket(text: str, buckets: int = 10_000) -> int:
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def stable_split(text: str, train_ratio: float) -> str:
    return "train" if stable_bucket(text) < int(train_ratio * 10_000) else "val"


def state_id(value: str) -> int:
    value = str(value or "").strip()
    if value in STATE_TO_ID:
        return STATE_TO_ID[value]
    if value in {"modal-replacement", "modal replacement", "modal_replacement"} or value.startswith("tampered"):
        return STATE_TO_ID[MODAL_REPLACEMENT]
    if "localized" in value:
        return STATE_TO_ID["localized-tamper"]
    if "content-unverified" in value:
        return STATE_TO_ID["content-unverified"]
    if value == LEGACY_REJECTED:
        return STATE_TO_ID[REJECTED]
    return STATE_TO_ID[REJECTED]


def token(*values: float) -> List[float]:
    vals = list(values[:FEAT_DIM])
    if len(vals) < FEAT_DIM:
        vals.extend([0.0] * (FEAT_DIM - len(vals)))
    return [float(v) for v in vals]


def log_norm(value: Any, scale: float = 10.0) -> float:
    return min(1.0, math.log1p(max(0.0, parse_float(value))) / math.log1p(scale))


def safe_json_list(value: str) -> List[float]:
    if not value:
        return []
    try:
        data = json.loads(value)
        if isinstance(data, list):
            return [parse_float(x) for x in data]
    except Exception:
        return []
    return []


def index_by_sample(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {str(r.get("sample_id", "")): r for r in rows if r.get("sample_id")}


def group_by_sample(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        sid = str(row.get("sample_id", ""))
        if sid:
            out[sid].append(row)
    return out


def best_audio_packet_by_sample(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    preferred = {"aac_128k": 0, "aac_320k": 1, "memory": 2}
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        sid = str(row.get("sample_id", ""))
        if not sid:
            continue
        rank = preferred.get(str(row.get("condition", "")), 99)
        old = out.get(sid)
        old_rank = preferred.get(str(old.get("condition", "")), 99) if old else 999
        if old is None or rank < old_rank:
            out[sid] = row
    return out


def add_claim_tokens(tokens: List[List[float]], types: List[int], loc: List[float], row: Dict[str, str]) -> Tuple[float, float, float]:
    sig = parse_bool(row.get("claim_signature_ok", row.get("signature_ok", "True")))
    scope = parse_bool(row.get("claim_source_scope_ok", row.get("source_scope_ok", "True")))
    claim = parse_bool(row.get("claim_final_ok", row.get("claim_recovered", "True")))
    candidate_count = row.get("claim_candidate_count", row.get("candidate_count_radius", "1"))
    tokens.append(token(0.0, 0.0, 0.0, 0.0, sig, scope, claim, log_norm(candidate_count, 10_000), parse_bool(row.get("claim_expected_hit", "True")), parse_bool(row.get("claim_enroll_scope_ok", "True")), 1.0))
    types.append(TOKEN_TYPES["claim"])
    loc.append(-1.0)
    return sig, scope, claim


def add_audio_carrier_token(tokens: List[List[float]], types: List[int], loc: List[float], row: Dict[str, str]) -> float:
    if not row:
        return -1.0
    total = max(1.0, parse_float(row.get("audio_total_windows", row.get("scan_windows", "1"))))
    valid = parse_float(row.get("audio_valid_packets", row.get("valid_packet_count", "0")))
    margin = parse_float(row.get("min_vote_margin", "0.0"))
    recovered = parse_bool(row.get("audio_tag_recovered", row.get("tag_recovered", "")))
    tokens.append(token(0.0, 0.0, 0.0, 0.0, recovered, valid / total, total / 96.0, margin / max(1.0, valid), parse_float(row.get("valid_packet_rate", valid / total)), parse_float(row.get("mean_score", 0.0)), parse_float(row.get("raw_bit_accuracy", 0.0)), parse_float(row.get("raw_bit_errors", 0.0)) / max(1.0, parse_float(row.get("tag_bits", 32.0))), 1.0))
    types.append(TOKEN_TYPES["audio_carrier"])
    loc.append(-1.0)
    return recovered


def add_video_carrier_token(tokens: List[List[float]], types: List[int], loc: List[float], row: Dict[str, str]) -> float:
    if not row:
        return -1.0
    recovered = parse_bool(row.get("video_pointer_recovered", row.get("m4_disambiguated_final_ok_proxy", row.get("exact_recovery", ""))))
    data_bits = max(1.0, parse_float(row.get("data_bits", "24")))
    tokens.append(token(0.0, 0.0, 0.0, 0.0, recovered, parse_float(row.get("video_vote_margin_min", row.get("vote_margin_min", 0.0))), parse_float(row.get("video_vote_margin_mean", row.get("vote_margin_mean", 0.0))), parse_float(row.get("voted_bit_accuracy", 1.0 - parse_float(row.get("voted_bit_errors", 0.0)) / data_bits)), parse_float(row.get("voted_bit_errors", 0.0)) / data_bits, parse_float(row.get("raw256_bit_errors", 0.0)) / 256.0, log_norm(row.get("candidate_count_radius", row.get("claim_candidate_count", 1)), 10_000), data_bits / 32.0, parse_float(row.get("video_frames_decoded", 0.0)) / 32.0, 1.0))
    types.append(TOKEN_TYPES["video_carrier"])
    loc.append(-1.0)
    return recovered


def add_audio_window_tokens(tokens: List[List[float]], types: List[int], loc: List[float], row: Dict[str, str], max_failed: int) -> float:
    if not row:
        return 0.0
    window_count = max(1.0, parse_float(row.get("window_count", "1")))
    failed = parse_float(row.get("failed_windows", 0.0))
    recovered = parse_float(row.get("recovered_windows", 0.0))
    failed_ratio = failed / window_count
    starts = safe_json_list(row.get("failed_window_starts_json", ""))
    tamper_signal = 1.0 if failed > 0 else 0.0
    tokens.append(token(0.0, 1.0, 0.0, 0.0, parse_float(row.get("recovery_ratio", 1.0)), failed_ratio, recovered / window_count, parse_float(row.get("longest_failed_run", 0.0)) / window_count, parse_float(row.get("max_ham_bits", 0.0)) / 255.0, parse_float(row.get("q90_ham_bits", 0.0)) / 255.0, tamper_signal, 1.0))
    types.append(TOKEN_TYPES["audio_window"])
    loc.append(tamper_signal)
    for i, start in enumerate(starts[:max_failed]):
        tokens.append(token(start / 20.0, (start + 0.5) / 20.0, i / max(1.0, max_failed), 0.0, 0.0, 1.0, failed_ratio, parse_float(row.get("max_ham_bits", 0.0)) / 255.0, 1.0, 1.0))
        types.append(TOKEN_TYPES["audio_window"])
        loc.append(1.0)
    return tamper_signal


def add_video_window_tokens(tokens: List[List[float]], types: List[int], loc: List[float], rows: List[Dict[str, str]], max_windows: int) -> float:
    tamper = 0.0
    if not rows:
        return tamper
    rows = sorted(rows, key=lambda r: parse_float(r.get("window_start_sec", 0.0)))[:max_windows]
    count = max(1.0, parse_float(rows[0].get("window_count", len(rows))))
    for row in rows:
        alarm = parse_bool(row.get("window_alarm"))
        tamper = max(tamper, alarm)
        tokens.append(token(parse_float(row.get("window_start_sec", 0.0)) / 20.0, parse_float(row.get("window_end_sec", 0.0)) / 20.0, parse_float(row.get("window_index", 0.0)) / count, 0.0, parse_float(row.get("window_score", 0.0)) / 255.0, alarm, parse_float(row.get("id_video_ham_bits", 0.0)) / 255.0, parse_float(row.get("evt_video_ham_bits", 0.0)) / 255.0, parse_bool(row.get("id_video_available", "True")), parse_bool(row.get("evt_video_available", "True")), alarm, 1.0))
        types.append(TOKEN_TYPES["video_window"])
        loc.append(alarm)
    return tamper


def add_semantic_summary_token(tokens: List[List[float]], types: List[int], loc: List[float], row: Dict[str, str], semantic_tamper: float) -> None:
    tokens.append(token(0.0, 1.0, 0.0, 0.0, 1.0 if str(row.get("state_audio", "")) == VERIFIED else 0.0, 1.0 if str(row.get("state_video", "")) == VERIFIED else 0.0, 1.0 if str(row.get("state_audio_sync", "")) == VERIFIED else 0.0, 1.0 if str(row.get("state_full_av", row.get("final_state", ""))) == VERIFIED else 0.0, semantic_tamper, parse_bool(row.get("m3_parity_when_signed_scope_ok", "True")), 1.0))
    types.append(TOKEN_TYPES["semantic_summary"])
    loc.append(-1.0)




def state_support(row: Dict[str, str], key: str, fallback: float = -1.0) -> float:
    if key not in row or row.get(key, "") == "":
        return fallback
    return 1.0 if str(row.get(key, "")) == VERIFIED else 0.0


def video_support_stats(rows: List[Dict[str, str]]) -> Dict[str, float]:
    if not rows:
        return {
            "row_count": 0.0,
            "window_count": 0.0,
            "coverage": 0.0,
            "id_available_rate": 0.0,
            "evt_available_rate": 0.0,
            "fallback_ratio": 1.0,
            "sufficient_support": 0.0,
            "missing_evidence": 1.0,
        }
    window_count = max(1.0, parse_float(rows[0].get("window_count", len(rows))))
    row_count = float(len(rows))
    coverage = min(1.0, row_count / window_count)
    id_rate = sum(parse_bool(r.get("id_video_available", "True")) for r in rows) / row_count
    evt_rate = sum(parse_bool(r.get("evt_video_available", "True")) for r in rows) / row_count
    fallback_ratio = 1.0 - min(id_rate, evt_rate)
    support = 1.0 if coverage >= 0.5 and evt_rate >= 0.5 else 0.0
    missing = 1.0 if coverage < 0.5 or evt_rate < 0.5 else 0.0
    return {
        "row_count": row_count,
        "window_count": window_count,
        "coverage": coverage,
        "id_available_rate": id_rate,
        "evt_available_rate": evt_rate,
        "fallback_ratio": fallback_ratio,
        "sufficient_support": support,
        "missing_evidence": missing,
    }


def add_sync_token(tokens: List[List[float]], types: List[int], loc: List[float], row: Dict[str, str]) -> float:
    support = state_support(row, "state_audio_sync", -1.0)
    if support < 0.0:
        support = 1.0 if str(row.get("state_full_av", row.get("final_state", ""))) == VERIFIED else 0.0
    tokens.append(token(0.0, 1.0, 0.0, 0.0, support, parse_bool(row.get("sync_available", "True")), parse_bool(row.get("m3_parity_when_signed_scope_ok", "True")), 1.0))
    types.append(TOKEN_TYPES["sync"])
    loc.append(-1.0)
    return support


def add_availability_token(
    tokens: List[List[float]],
    types: List[int],
    loc: List[float],
    audio_carrier_row: Dict[str, str],
    video_carrier_row: Dict[str, str],
    audio_window_row: Dict[str, str],
    video_stats: Dict[str, float],
) -> float:
    has_audio_carrier = 1.0 if audio_carrier_row else 0.0
    has_video_carrier = 1.0 if video_carrier_row else 0.0
    audio_window_count = parse_float(audio_window_row.get("window_count", 0.0)) if audio_window_row else 0.0
    missing = max(
        0.0 if audio_window_count > 0.0 else 1.0,
        parse_float(video_stats.get("missing_evidence", 1.0)),
    )
    tokens.append(token(
        0.0,
        0.0,
        1.0,
        0.0,
        has_audio_carrier,
        has_video_carrier,
        min(1.0, audio_window_count / 64.0),
        parse_float(video_stats.get("coverage", 0.0)),
        parse_float(video_stats.get("id_available_rate", 0.0)),
        parse_float(video_stats.get("evt_available_rate", 0.0)),
        parse_float(video_stats.get("fallback_ratio", 1.0)),
        missing,
        parse_float(video_stats.get("sufficient_support", 0.0)),
        1.0,
    ))
    types.append(TOKEN_TYPES["availability"])
    loc.append(-1.0)
    return missing


def build_predicate_labels(
    row: Dict[str, str],
    source: str,
    label: int,
    sig: float,
    scope: float,
    claim: float,
    audio_label: float,
    video_label: float,
    sync_label: float,
    audio_tamper: float,
    video_tamper: float,
    semantic_tamper: float,
    missing_evidence: float,
    video_stats: Dict[str, float],
) -> List[float]:
    audio_content = state_support(row, "state_audio", 1.0 if label == STATE_TO_ID[VERIFIED] else 0.0)
    video_content = state_support(row, "state_video", 1.0 if label == STATE_TO_ID[VERIFIED] else 0.0)
    if source == "m7_carrier_val":
        # Carrier-validation rows supervise pointer reliability, not full M4 semantic coverage.
        missing_evidence = 0.0
        if video_label >= 0.0:
            video_content = video_label
        if audio_label >= 0.0:
            audio_content = audio_label
    local_tamper = max(audio_tamper, video_tamper, 1.0 if label in {STATE_TO_ID["localized-tamper"], STATE_TO_ID[MODAL_REPLACEMENT]} else 0.0)
    modal_conflict = 1.0 if audio_content >= 0.0 and video_content >= 0.0 and abs(audio_content - video_content) > 0.5 else 0.0
    sufficient_video = parse_float(video_stats.get("sufficient_support", 0.0))
    if source == "m7_carrier_val" and video_label >= 0.0:
        sufficient_video = video_label
    return [
        sig * claim,
        scope,
        audio_label,
        video_label,
        audio_content,
        video_content,
        sync_label,
        local_tamper,
        missing_evidence,
        modal_conflict,
        sufficient_video,
    ]

def make_example(row: Dict[str, str], source: str, label_state: str, claim_row: Dict[str, str], signed_row: Dict[str, str], audio_carrier_row: Dict[str, str], video_carrier_row: Dict[str, str], audio_window_row: Dict[str, str], video_window_rows: List[Dict[str, str]], max_video_windows: int, max_audio_failed_windows: int) -> Example:
    tokens: List[List[float]] = []
    types: List[int] = []
    loc: List[float] = []
    merged_claim = {**row, **signed_row, **claim_row}
    merged_semantic = {**row, **signed_row}
    merged_audio_carrier = {**audio_carrier_row, **claim_row}
    merged_video_carrier = {**video_carrier_row, **claim_row}
    sig, scope, claim = add_claim_tokens(tokens, types, loc, merged_claim)
    audio_label = add_audio_carrier_token(tokens, types, loc, merged_audio_carrier)
    video_label = add_video_carrier_token(tokens, types, loc, merged_video_carrier)
    audio_tamper = add_audio_window_tokens(tokens, types, loc, audio_window_row, max_audio_failed_windows)
    video_tamper = add_video_window_tokens(tokens, types, loc, video_window_rows, max_video_windows)
    video_stats = video_support_stats(video_window_rows)
    missing_evidence = add_availability_token(tokens, types, loc, merged_audio_carrier, merged_video_carrier, audio_window_row, video_stats)
    sync_label = add_sync_token(tokens, types, loc, merged_semantic)
    sid = str(row.get("sample_id", claim_row.get("sample_id", "")))
    label = state_id(label_state)
    semantic_tamper = 0.0 if label == STATE_TO_ID[VERIFIED] else 1.0
    if source.startswith("signed"):
        semantic_tamper = max(semantic_tamper, audio_tamper, video_tamper)
    add_semantic_summary_token(tokens, types, loc, merged_semantic, semantic_tamper)
    predicate_labels = build_predicate_labels(
        merged_semantic,
        source,
        label,
        sig,
        scope,
        claim,
        audio_label,
        video_label,
        sync_label,
        audio_tamper,
        video_tamper,
        semantic_tamper,
        missing_evidence,
        video_stats,
    )
    return Example(sid, source, tokens, types, loc, label, audio_label, video_label, sig, scope, claim, semantic_tamper, predicate_labels)


def build_examples(args: argparse.Namespace) -> Tuple[List[Example], Dict[str, Any]]:
    audio_packet_rows = read_many_csv(args.audio_packet_rows)
    video_pointer_rows = read_csv(args.video_pointer_rows)
    m3_rows = read_csv(args.m3_fused)
    signed_rows = read_csv(args.signed_m4_rows)
    claim_rows = read_csv(args.claim_rows)
    audio_window_rows = read_csv(args.audio_window_scores)
    video_window_rows = read_csv(args.video_window_scores)
    claim_by_sample = index_by_sample(claim_rows)
    signed_by_sample = index_by_sample(signed_rows)
    audio_packet_by_sample = best_audio_packet_by_sample(audio_packet_rows)
    audio_window_by_sample = index_by_sample(audio_window_rows)
    video_window_by_sample = group_by_sample(video_window_rows)
    examples: List[Example] = []
    for row in m3_rows:
        sid = str(row.get("sample_id", ""))
        if not sid:
            continue
        examples.append(make_example(row, "m3_windowed_semantic", str(row.get("state_full_av", REJECTED)), claim_by_sample.get(sid, {}), signed_by_sample.get(sid, {}), {}, {}, audio_window_by_sample.get(sid, {}), video_window_by_sample.get(sid, []), args.max_video_windows, args.max_audio_failed_windows))
    video_added = 0
    for row in video_pointer_rows:
        if args.limit_carrier > 0 and video_added >= args.limit_carrier:
            break
        if str(row.get("data_bits", "")) not in {"24", "32"}:
            continue
        sid = str(row.get("sample_id", ""))
        audio = audio_packet_by_sample.get(sid, {})
        claim_ok = bool(parse_bool(row.get("m4_disambiguated_final_ok_proxy")) or parse_bool(row.get("top1_signature_ok")) or parse_bool(audio.get("tag_recovered")))
        label_state = VERIFIED if claim_ok else REJECTED
        pseudo = {
            "sample_id": sid,
            "state_full_av": label_state,
            "state_audio": label_state,
            "state_video": label_state,
            "state_audio_sync": label_state,
            "claim_signature_ok": row.get("top1_signature_ok", str(claim_ok)),
            "claim_source_scope_ok": row.get("top1_source_scope_ok", str(claim_ok)),
            "claim_final_ok": str(claim_ok),
            "claim_candidate_count": row.get("candidate_count_radius", "1"),
        }
        examples.append(make_example(pseudo, "m7_carrier_val", label_state, pseudo, {}, audio, row, {}, [], args.max_video_windows, args.max_audio_failed_windows))
        video_added += 1
    if args.limit_examples > 0:
        examples = examples[:args.limit_examples]
    meta = {
        "m3_rows": len(m3_rows),
        "signed_rows": len(signed_rows),
        "claim_rows": len(claim_rows),
        "audio_packet_rows": len(audio_packet_rows),
        "video_pointer_rows": len(video_pointer_rows),
        "examples": len(examples),
        "state_distribution": dict(Counter(STATES[e.state_label] for e in examples)),
        "source_distribution": dict(Counter(e.source for e in examples)),
    }
    return examples, meta


def state_mask_for(ex: Example) -> np.ndarray:
    mask = np.ones(len(STATES), dtype=np.bool_)
    if ex.signature_ok < 0.5 or ex.source_scope_ok < 0.5 or ex.claim_recovered < 0.5:
        mask[STATE_TO_ID[VERIFIED]] = False
        mask[STATE_TO_ID["source-only verified"]] = False
    if ex.semantic_tamper >= 0.5:
        mask[STATE_TO_ID[VERIFIED]] = False
    mask[ex.state_label] = True
    return mask


def materialize_dataset(examples: List[Example], max_tokens: int, train_ratio: float) -> Dict[str, Any]:
    n = len(examples)
    tokens = np.zeros((n, max_tokens, FEAT_DIM), dtype=np.float32)
    token_types = np.zeros((n, max_tokens), dtype=np.int64)
    token_mask = np.zeros((n, max_tokens), dtype=np.bool_)
    loc_labels = np.full((n, max_tokens), -1.0, dtype=np.float32)
    state_labels = np.zeros(n, dtype=np.int64)
    audio_labels = np.full(n, -1.0, dtype=np.float32)
    video_labels = np.full(n, -1.0, dtype=np.float32)
    predicate_labels = np.full((n, len(PREDICATES)), -1.0, dtype=np.float32)
    state_masks = np.zeros((n, len(STATES)), dtype=np.bool_)
    sample_ids: List[str] = []
    sources: List[str] = []
    for i, ex in enumerate(examples):
        keep = min(max_tokens, len(ex.tokens))
        tokens[i, :keep] = np.asarray(ex.tokens[:keep], dtype=np.float32)
        token_types[i, :keep] = np.asarray(ex.token_types[:keep], dtype=np.int64)
        token_mask[i, :keep] = True
        loc_labels[i, :keep] = np.asarray(ex.loc_labels[:keep], dtype=np.float32)
        state_labels[i] = ex.state_label
        audio_labels[i] = ex.audio_label
        video_labels[i] = ex.video_label
        predicate_labels[i] = np.asarray(ex.predicate_labels, dtype=np.float32)
        state_masks[i] = state_mask_for(ex)
        sample_ids.append(ex.sample_id)
        sources.append(ex.source)
    split = np.asarray([stable_split(sid, train_ratio) for sid in sample_ids])
    return {"tokens": tokens, "token_types": token_types, "token_mask": token_mask, "loc_labels": loc_labels, "state_labels": state_labels, "audio_labels": audio_labels, "video_labels": video_labels, "predicate_labels": predicate_labels, "state_masks": state_masks, "sample_ids": np.asarray(sample_ids), "sources": np.asarray(sources), "split": split}


def normalize_tokens(data: Dict[str, Any]) -> Dict[str, Any]:
    train = data["split"] == "train"
    valid = data["token_mask"] & train[:, None]
    vals = data["tokens"][valid]
    if vals.size == 0:
        mean = np.zeros((1, 1, FEAT_DIM), dtype=np.float32)
        std = np.ones((1, 1, FEAT_DIM), dtype=np.float32)
    else:
        mean = vals.mean(axis=0, keepdims=True).reshape(1, 1, FEAT_DIM).astype(np.float32)
        std = vals.std(axis=0, keepdims=True).reshape(1, 1, FEAT_DIM).astype(np.float32)
        std[std < 1e-6] = 1.0
    data["tokens"] = ((data["tokens"] - mean) / std).astype(np.float32)
    data["normalization_mean"] = mean.reshape(-1)
    data["normalization_std"] = std.reshape(-1)
    return data


class ContextEncoderLayer(nn.Module):
    """Build contextual token features before dependency-constrained reasoning."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor, token_types: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        y = self.norm1(x)
        qkv = self.qkv(y).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~token_mask[:, None, None, :], -1.0e4)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, seq_len, self.hidden_dim)
        x = x + self.dropout(self.out_proj(ctx))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class SparseDependencyResidual(nn.Module):
    """Route messages only along declared provenance-evidence dependencies."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_types = len(TOKEN_TYPES)
        self.norm = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.relation_bias = nn.Parameter(
            torch.zeros(num_heads, self.num_types, self.num_types)
        )
        self.mix_logit = nn.Parameter(torch.tensor(-2.0))
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.register_buffer(
            "type_adjacency", self._build_type_adjacency(), persistent=False
        )

    @staticmethod
    def _build_type_adjacency() -> torch.Tensor:
        adjacency = torch.eye(len(TOKEN_TYPES), dtype=torch.bool)

        def connect(left: str, right: str) -> None:
            left_id = TOKEN_TYPES[left]
            right_id = TOKEN_TYPES[right]
            adjacency[left_id, right_id] = True
            adjacency[right_id, left_id] = True

        connect("claim", "audio_carrier")
        connect("claim", "video_carrier")
        connect("claim", "semantic_summary")
        connect("claim", "sync")
        connect("claim", "availability")
        connect("audio_carrier", "audio_window")
        connect("video_carrier", "video_window")
        connect("audio_carrier", "semantic_summary")
        connect("video_carrier", "semantic_summary")
        connect("audio_carrier", "availability")
        connect("video_carrier", "availability")
        connect("audio_window", "semantic_summary")
        connect("video_window", "semantic_summary")
        connect("audio_window", "sync")
        connect("video_window", "sync")
        connect("audio_window", "availability")
        connect("video_window", "availability")
        connect("semantic_summary", "sync")
        connect("semantic_summary", "availability")
        connect("sync", "availability")
        return adjacency

    def _edge_mask(
        self, token_types: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        valid = token_mask.unsqueeze(2) & token_mask.unsqueeze(1)
        query_types = token_types.unsqueeze(2)
        key_types = token_types.unsqueeze(1)
        edges = self.type_adjacency[query_types, key_types]

        positions = torch.arange(token_types.shape[1], device=token_types.device)
        distance = (positions[:, None] - positions[None, :]).abs()
        for window_type in (
            TOKEN_TYPES["audio_window"],
            TOKEN_TYPES["video_window"],
        ):
            same_window_type = (query_types == window_type) & (
                key_types == window_type
            )
            edges = edges & ~(same_window_type & (distance[None, :, :] > 1))
        return edges & valid

    def forward(
        self,
        x: torch.Tensor,
        token_types: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, token_count, _ = x.shape
        y = self.norm(x)
        qkv = self.qkv(y).view(
            batch_size, token_count, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        relation_index = (
            token_types.unsqueeze(2) * self.num_types + token_types.unsqueeze(1)
        )
        relation_flat = self.relation_bias.reshape(
            self.num_heads, self.num_types * self.num_types
        )
        relation = relation_flat[:, relation_index.reshape(batch_size, -1)]
        relation = relation.permute(1, 0, 2).reshape(
            batch_size, self.num_heads, token_count, token_count
        )
        scores = scores + relation
        edges = self._edge_mask(token_types, token_mask)
        scores = scores.masked_fill(~edges[:, None, :, :], -1.0e4)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attention, value)
        context = context.transpose(1, 2).reshape(
            batch_size, token_count, self.hidden_dim
        )
        gate = torch.sigmoid(self.mix_logit)
        return x + gate * self.dropout(self.out_proj(context))


class HierarchicalEvidenceGraph(nn.Module):
    """Final evidence verifier with token- and type-level graph reasoning."""

    def __init__(self, feat_dim: int, hidden_dim: int, num_heads: int, layers: int, dropout: float, max_tokens: int):
        super().__init__()
        self.input = nn.Linear(feat_dim, hidden_dim)
        self.type_emb = nn.Embedding(len(TOKEN_TYPES), hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(max_tokens, hidden_dim))
        self.context_encoder = nn.ModuleList(
            [ContextEncoderLayer(hidden_dim, num_heads, dropout) for _ in range(layers)]
        )
        self.token_graph = SparseDependencyResidual(hidden_dim, num_heads, dropout)
        self.slot_graph = SparseDependencyResidual(hidden_dim, num_heads, dropout)
        self.slot_norm = nn.LayerNorm(hidden_dim)
        self.slot_readout = nn.Sequential(
            nn.LayerNorm(hidden_dim * len(TOKEN_TYPES) + len(TOKEN_TYPES)),
            nn.Linear(
                hidden_dim * len(TOKEN_TYPES) + len(TOKEN_TYPES), hidden_dim * 2
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.slot_mix_logit = nn.Parameter(torch.tensor(-2.0))
        self.norm = nn.LayerNorm(hidden_dim)
        self.direct_state_head = nn.Linear(hidden_dim, len(STATES))
        self.predicate_head = nn.Linear(hidden_dim, len(PREDICATES))
        self.audio_head = nn.Linear(hidden_dim, 1)
        self.video_head = nn.Linear(hidden_dim, 1)
        self.loc_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.slot_readout[-1].weight)
        nn.init.zeros_(self.slot_readout[-1].bias)

    def compose_state_logits(self, direct_logits: torch.Tensor, predicate_logits: torch.Tensor) -> torch.Tensor:
        p = {name: predicate_logits[:, idx] for name, idx in PRED_TO_ID.items()}
        carrier = torch.maximum(p["audio_carrier_reliable"], p["video_carrier_reliable"])
        positive = torch.stack([
            p["claim_authentic"],
            p["source_scope_valid"],
            carrier,
            p["audio_content_consistent"],
            p["video_content_consistent"],
            p["sync_consistent"],
            p["sufficient_video_support"],
        ], dim=0).mean(dim=0)
        risk = torch.stack([
            p["local_tamper"],
            p["missing_evidence"],
            p["modal_conflict"],
            -carrier,
        ], dim=0).amax(dim=0)
        source_support = torch.stack([
            p["claim_authentic"],
            p["source_scope_valid"],
            carrier,
        ], dim=0).mean(dim=0)
        source_risk = torch.stack([p["modal_conflict"], -carrier], dim=0).amax(dim=0)
        logits = direct_logits.clone()
        # Verified is admitted by positive support, not by absence of attack logits.
        logits[:, STATE_TO_ID[VERIFIED]] = positive - risk
        logits[:, STATE_TO_ID["source-only verified"]] = source_support - source_risk
        return logits

    def _hierarchical_readout(
        self,
        hidden: torch.Tensor,
        token_types: torch.Tensor,
        token_mask: torch.Tensor,
        global_pool: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = hidden.shape[0]
        type_ids = torch.arange(len(TOKEN_TYPES), device=hidden.device)
        slot_mask = (
            token_types.unsqueeze(-1) == type_ids.view(1, 1, -1)
        ) & token_mask.unsqueeze(-1)
        slot_count = slot_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        slots = torch.einsum(
            "bth,btn->bnh", hidden, slot_mask.to(hidden.dtype)
        ) / slot_count
        slot_present = slot_mask.any(dim=1)
        slot_types = type_ids.unsqueeze(0).expand(batch_size, -1)
        slots = slots + self.type_emb(slot_types)
        slots = self.slot_graph(slots, slot_types, slot_present)
        slots = self.slot_norm(slots)
        slots = slots * slot_present.unsqueeze(-1).to(slots.dtype)
        structured = self.slot_readout(
            torch.cat(
                [slots.flatten(start_dim=1), slot_present.to(slots.dtype)], dim=1
            )
        )
        return global_pool + torch.sigmoid(self.slot_mix_logit) * structured

    def forward(self, tokens: torch.Tensor, token_types: torch.Tensor, token_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, t, _ = tokens.shape
        x = self.input(tokens) + self.pos_emb[:t].unsqueeze(0)
        for layer in self.context_encoder:
            x = layer(x, token_types, token_mask)
        x = self.token_graph(x, token_types, token_mask)
        h = self.norm(x)
        denom = token_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        global_pool = (h * token_mask.unsqueeze(-1).float()).sum(dim=1) / denom
        pooled = self._hierarchical_readout(
            h, token_types, token_mask, global_pool
        )
        direct_state_logits = self.direct_state_head(pooled)
        predicate_logits = self.predicate_head(pooled)
        state_logits = self.compose_state_logits(direct_state_logits, predicate_logits)
        return {
            "state_logits": state_logits,
            "direct_state_logits": direct_state_logits,
            "predicate_logits": predicate_logits,
            "audio_logits": self.audio_head(pooled).squeeze(-1),
            "video_logits": self.video_head(pooled).squeeze(-1),
            "loc_logits": self.loc_head(h).squeeze(-1),
        }


def tensor_slice(data: Dict[str, Any], idx: np.ndarray, device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(data[k][idx]).to(device) for k in ["tokens", "token_types", "token_mask", "loc_labels", "state_labels", "audio_labels", "video_labels", "predicate_labels", "state_masks"]}


def masked_state_logits(logits: torch.Tensor, state_masks: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~state_masks, -1.0e4)


def availability_constrained_state_masks(
    state_masks: torch.Tensor,
    claim_available: torch.Tensor,
    audio_carrier_available: torch.Tensor,
    video_carrier_available: torch.Tensor,
    audio_content_available: torch.Tensor,
    video_content_available: torch.Tensor,
    sync_available: torch.Tensor,
) -> torch.Tensor:
    """Apply positive-support constraints derived from evidence availability.

    Either carrier may recover the registry pointer. Full content verification
    additionally requires claim, audio-content, video-content, and sync support.
    """
    constrained = state_masks.clone()
    anchor_available = claim_available & (
        audio_carrier_available | video_carrier_available
    )
    content_support = (
        anchor_available
        & audio_content_available
        & video_content_available
        & sync_available
    )
    constrained[~anchor_available, STATE_TO_ID[VERIFIED]] = False
    constrained[~anchor_available, STATE_TO_ID["source-only verified"]] = False
    constrained[~content_support, STATE_TO_ID[VERIFIED]] = False
    return constrained


def masked_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = labels >= 0.0
    if not mask.any():
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(logits[mask], labels[mask])


def loc_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = labels >= 0.0
    if not mask.any():
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(logits[mask], labels[mask])


def predicate_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = labels >= 0.0
    if not mask.any():
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(logits[mask], labels[mask])


def predicate_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if logits.size == 0:
        return out
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = probs >= 0.5
    for name, idx in PRED_TO_ID.items():
        mask = labels[:, idx] >= 0.0
        if not mask.any():
            continue
        y = labels[mask, idx] >= 0.5
        p = pred[mask, idx]
        out[name] = {
            "n": int(mask.sum()),
            "accuracy": float((p == y).mean()),
            "fp": int((p & ~y).sum()),
            "fn": int((~p & y).sum()),
        }
    return out


def apply_counterfactual_dropout(
    batch: Dict[str, torch.Tensor],
    drop_prob: float,
    normalized_zero: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    if drop_prob <= 0.0:
        return batch
    tokens = batch["tokens"].clone()
    token_mask = batch["token_mask"].clone()
    state_labels = batch["state_labels"].clone()
    state_masks = batch["state_masks"].clone()
    predicate_labels = batch["predicate_labels"].clone()
    loc_labels = batch["loc_labels"].clone()
    token_types = batch["token_types"]
    if normalized_zero is None:
        normalized_zero = torch.zeros(tokens.shape[-1], device=tokens.device, dtype=tokens.dtype)
    else:
        normalized_zero = normalized_zero.to(device=tokens.device, dtype=tokens.dtype)

    # Five evidence-drop operations plus two provenance-specific counterfactuals.
    # Keeping one outer probability preserves the total augmentation rate.
    operation_count = 7
    choose = torch.rand(tokens.shape[0], device=tokens.device) < drop_prob
    if not choose.any():
        out = dict(batch)
        out["tokens"] = tokens
        out["token_mask"] = token_mask
        out["state_labels"] = state_labels
        out["state_masks"] = state_masks
        out["predicate_labels"] = predicate_labels
        out["loc_labels"] = loc_labels
        return out
    picked = torch.randint(0, operation_count, (tokens.shape[0],), device=tokens.device)
    drop_types = [
        TOKEN_TYPES["claim"],
        TOKEN_TYPES["audio_carrier"],
        TOKEN_TYPES["video_carrier"],
        TOKEN_TYPES["video_window"],
        TOKEN_TYPES["sync"],
    ]
    rows_with_video_windows = torch.where(
        ((token_types == TOKEN_TYPES["video_window"]) & token_mask).any(dim=1)
    )[0]

    def downgrade_verified(row: torch.Tensor, target_state: str) -> None:
        if state_labels[row] != STATE_TO_ID[VERIFIED]:
            return
        state_labels[row] = STATE_TO_ID[target_state]
        state_masks[row, STATE_TO_ID[VERIFIED]] = False
        state_masks[row, STATE_TO_ID["source-only verified"]] = False
        state_masks[row, STATE_TO_ID[target_state]] = True

    for row in torch.where(choose)[0]:
        operation = int(picked[row].item())
        if operation < len(drop_types):
            drop_type = drop_types[operation]
            drop = (token_types[row] == drop_type) & token_mask[row]
            token_mask[row, drop] = False
            tokens[row, drop] = 0.0
            downgrade_verified(row, "content-unverified")
            if drop_type == TOKEN_TYPES["claim"]:
                predicate_labels[row, PRED_TO_ID["claim_authentic"]] = 0.0
            elif drop_type == TOKEN_TYPES["audio_carrier"]:
                predicate_labels[row, PRED_TO_ID["audio_carrier_reliable"]] = 0.0
            elif drop_type == TOKEN_TYPES["video_carrier"]:
                predicate_labels[row, PRED_TO_ID["video_carrier_reliable"]] = 0.0
            elif drop_type == TOKEN_TYPES["video_window"]:
                predicate_labels[row, PRED_TO_ID["sufficient_video_support"]] = 0.0
                predicate_labels[row, PRED_TO_ID["missing_evidence"]] = 1.0
            elif drop_type == TOKEN_TYPES["sync"]:
                predicate_labels[row, PRED_TO_ID["sync_consistent"]] = 0.0
            continue

        if operation == 5:
            # Keep the claim token available while invalidating its signed
            # authenticity, source-scope, and recovery support fields.
            claim = (token_types[row] == TOKEN_TYPES["claim"]) & token_mask[row]
            tokens[row, claim, 4:7] = normalized_zero[4:7]
            downgrade_verified(row, REJECTED)
            predicate_labels[row, PRED_TO_ID["claim_authentic"]] = 0.0
            predicate_labels[row, PRED_TO_ID["source_scope_valid"]] = 0.0
            continue

        # Copy semantic window channels from another sample while retaining
        # the target timeline coordinates.
        target = torch.where(
            (token_types[row] == TOKEN_TYPES["video_window"]) & token_mask[row]
        )[0]
        if target.numel() == 0 or rows_with_video_windows.numel() < 2:
            continue
        donor_candidates = rows_with_video_windows[rows_with_video_windows != row]
        if donor_candidates.numel() == 0:
            continue
        donor = donor_candidates[
            torch.randint(0, donor_candidates.numel(), (1,), device=tokens.device)[0]
        ]
        donor_pos = torch.where(
            (token_types[donor] == TOKEN_TYPES["video_window"]) & token_mask[donor]
        )[0]
        # An earlier counterfactual operation in this batch may have masked
        # all windows of a row that was eligible when the donor pool was built.
        if donor_pos.numel() == 0:
            continue
        donor_pick = donor_pos[
            torch.arange(target.numel(), device=tokens.device) % donor_pos.numel()
        ]
        tokens[row, target, 4:] = tokens[donor, donor_pick, 4:].detach()
        loc_labels[row, target] = 1.0
        downgrade_verified(row, "localized-tamper")
        predicate_labels[row, PRED_TO_ID["video_content_consistent"]] = 0.0
        predicate_labels[row, PRED_TO_ID["local_tamper"]] = 1.0
        predicate_labels[row, PRED_TO_ID["modal_conflict"]] = 1.0
        predicate_labels[row, PRED_TO_ID["sufficient_video_support"]] = 0.0
    out = dict(batch)
    out["tokens"] = tokens
    out["token_mask"] = token_mask
    out["state_labels"] = state_labels
    out["state_masks"] = state_masks
    out["predicate_labels"] = predicate_labels
    out["loc_labels"] = loc_labels
    return out


def compute_metrics(logits: np.ndarray, masked_logits: np.ndarray, labels: np.ndarray, sources: np.ndarray, direct_logits: np.ndarray | None = None) -> Dict[str, Any]:
    raw_pred = logits.argmax(axis=1)
    pred = masked_logits.argmax(axis=1)
    unsafe = labels != STATE_TO_ID[VERIFIED]
    tamper_source = np.asarray([s == "m3_windowed_semantic" for s in sources])
    probs = torch.softmax(torch.from_numpy(masked_logits), dim=1).numpy()
    raw_probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    out = {
        "n": int(labels.size),
        "accuracy": float((pred == labels).mean()) if labels.size else None,
        "raw_accuracy_without_safety_mask": float((raw_pred == labels).mean()) if labels.size else None,
        "verified_pred": int((pred == STATE_TO_ID[VERIFIED]).sum()),
        "rejected_pred": int((pred == STATE_TO_ID[REJECTED]).sum()),
        "unsafe_false_verified": int(((pred == STATE_TO_ID[VERIFIED]) & unsafe).sum()),
        "raw_unsafe_false_verified_without_safety_mask": int(((raw_pred == STATE_TO_ID[VERIFIED]) & unsafe).sum()),
        "tamper_false_verified": int(((pred == STATE_TO_ID[VERIFIED]) & unsafe & tamper_source).sum()),
        "raw_tamper_false_verified_without_safety_mask": int(((raw_pred == STATE_TO_ID[VERIFIED]) & unsafe & tamper_source).sum()),
        "brier_state": float(np.mean((probs - np.eye(len(STATES))[labels]) ** 2)) if labels.size else None,
        "raw_brier_state_without_safety_mask": float(np.mean((raw_probs - np.eye(len(STATES))[labels]) ** 2)) if labels.size else None,
        "label_distribution": dict(Counter(STATES[int(x)] for x in labels.tolist())),
        "prediction_distribution": dict(Counter(STATES[int(x)] for x in pred.tolist())),
        "raw_prediction_distribution_without_safety_mask": dict(Counter(STATES[int(x)] for x in raw_pred.tolist())),
    }
    if direct_logits is not None:
        direct_pred = direct_logits.argmax(axis=1)
        direct_probs = torch.softmax(torch.from_numpy(direct_logits), dim=1).numpy()
        out.update({
            "direct_accuracy_without_predicate_logic": float((direct_pred == labels).mean()) if labels.size else None,
            "direct_unsafe_false_verified_without_predicate_logic": int(((direct_pred == STATE_TO_ID[VERIFIED]) & unsafe).sum()),
            "direct_tamper_false_verified_without_predicate_logic": int(((direct_pred == STATE_TO_ID[VERIFIED]) & unsafe & tamper_source).sum()),
            "direct_brier_state_without_predicate_logic": float(np.mean((direct_probs - np.eye(len(STATES))[labels]) ** 2)) if labels.size else None,
            "direct_prediction_distribution_without_predicate_logic": dict(Counter(STATES[int(x)] for x in direct_pred.tolist())),
        })
    return out


def prediction_rows(data: Dict[str, Any], idx: np.ndarray, logits: np.ndarray, masked_logits: np.ndarray, direct_logits: np.ndarray | None = None) -> List[Dict[str, Any]]:
    raw_pred = logits.argmax(axis=1)
    pred = masked_logits.argmax(axis=1)
    direct_pred = direct_logits.argmax(axis=1) if direct_logits is not None else raw_pred
    rows = []
    for local_i, global_i in enumerate(idx[:200]):
        rows.append({
            "sample_id": str(data["sample_ids"][global_i]),
            "source": str(data["sources"][global_i]),
            "label": STATES[int(data["state_labels"][global_i])],
            "pred": STATES[int(pred[local_i])],
            "raw_pred_without_safety_mask": STATES[int(raw_pred[local_i])],
            "direct_pred_without_predicate_logic": STATES[int(direct_pred[local_i])],
        })
    return rows


def run_eval(model: HierarchicalEvidenceGraph, data: Dict[str, Any], idx: np.ndarray, device: torch.device, batch_size: int) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    model.eval()
    logits_all: List[np.ndarray] = []
    masked_all: List[np.ndarray] = []
    direct_all: List[np.ndarray] = []
    predicate_all: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start:start + batch_size]
            batch = tensor_slice(data, batch_idx, device)
            out = model(batch["tokens"], batch["token_types"], batch["token_mask"])
            raw = out["state_logits"]
            masked = masked_state_logits(raw, batch["state_masks"])
            logits_all.append(raw.detach().cpu().numpy())
            masked_all.append(masked.detach().cpu().numpy())
            direct_all.append(out["direct_state_logits"].detach().cpu().numpy())
            predicate_all.append(out["predicate_logits"].detach().cpu().numpy())
    logits = np.concatenate(logits_all, axis=0) if logits_all else np.zeros((0, len(STATES)), dtype=np.float32)
    masked = np.concatenate(masked_all, axis=0) if masked_all else np.zeros((0, len(STATES)), dtype=np.float32)
    direct = np.concatenate(direct_all, axis=0) if direct_all else np.zeros((0, len(STATES)), dtype=np.float32)
    predicates = np.concatenate(predicate_all, axis=0) if predicate_all else np.zeros((0, len(PREDICATES)), dtype=np.float32)
    labels = data["state_labels"][idx]
    sources = data["sources"][idx]
    metrics = compute_metrics(logits, masked, labels, sources, direct)
    metrics["predicate_metrics"] = predicate_metrics(predicates, data["predicate_labels"][idx])
    return metrics, {"logits": logits, "masked_logits": masked, "direct_logits": direct, "predicate_logits": predicates}


def train(data: Dict[str, Any], args: argparse.Namespace) -> Tuple[HierarchicalEvidenceGraph, Dict[str, Any]]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    train_idx = np.where(data["split"] == "train")[0]
    val_idx = np.where(data["split"] != "train")[0]
    if train_idx.size == 0 or val_idx.size == 0:
        perm = np.random.default_rng(args.seed).permutation(len(data["state_labels"]))
        cut = max(1, int(args.train_ratio * len(perm)))
        train_idx, val_idx = perm[:cut], perm[cut:]
    model = HierarchicalEvidenceGraph(
        FEAT_DIM,
        args.hidden_dim,
        args.num_heads,
        args.layers,
        args.dropout,
        args.max_tokens,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_indices = np.asarray(train_idx)
    rng = np.random.default_rng(args.seed)
    normalized_zero = torch.from_numpy(
        (-data["normalization_mean"] / data["normalization_std"]).astype(np.float32)
    ).to(device)
    for _epoch in range(max(1, args.epochs)):
        rng.shuffle(train_indices)
        model.train()
        for start in range(0, len(train_indices), args.batch_size):
            idx = train_indices[start:start + args.batch_size]
            batch = tensor_slice(data, idx, device)
            batch = apply_counterfactual_dropout(
                batch,
                args.counterfactual_dropout,
                normalized_zero=normalized_zero,
            )
            out = model(batch["tokens"], batch["token_types"], batch["token_mask"])
            masked_logits = masked_state_logits(out["state_logits"], batch["state_masks"])
            state = nn.functional.cross_entropy(masked_logits, batch["state_labels"])
            direct_masked_logits = masked_state_logits(out["direct_state_logits"], batch["state_masks"])
            direct_state = nn.functional.cross_entropy(direct_masked_logits, batch["state_labels"])
            audio = masked_bce(out["audio_logits"], batch["audio_labels"])
            video = masked_bce(out["video_logits"], batch["video_labels"])
            local = loc_loss(out["loc_logits"], batch["loc_labels"])
            predicates = predicate_loss(out["predicate_logits"], batch["predicate_labels"])
            probs = torch.softmax(masked_logits, dim=1)
            unsafe = batch["state_labels"] != STATE_TO_ID[VERIFIED]
            safety = -torch.log1p(-probs[unsafe, STATE_TO_ID[VERIFIED]].clamp(max=1.0 - 1e-6)).mean() if unsafe.any() else state * 0.0
            y_onehot = nn.functional.one_hot(batch["state_labels"], num_classes=len(STATES)).float()
            calib = ((probs - y_onehot) ** 2).mean()
            loss = state + args.direct_state_loss_weight * direct_state + args.audio_loss_weight * audio + args.video_loss_weight * video + args.localization_loss_weight * local + args.predicate_loss_weight * predicates + args.safety_loss_weight * safety + args.calibration_loss_weight * calib
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    train_metrics, _ = run_eval(model, data, train_idx, device, args.batch_size)
    val_metrics, val_raw = run_eval(model, data, val_idx, device, args.batch_size)
    return model, {"device": str(device), "train_rows": int(train_idx.size), "val_rows": int(val_idx.size), "train": train_metrics, "val": val_metrics, "loss_weights": {"direct_state": args.direct_state_loss_weight, "audio": args.audio_loss_weight, "video": args.video_loss_weight, "localization": args.localization_loss_weight, "predicate": args.predicate_loss_weight, "safety": args.safety_loss_weight, "calibration": args.calibration_loss_weight, "counterfactual_dropout": args.counterfactual_dropout}, "val_predictions": prediction_rows(data, val_idx, val_raw["logits"], val_raw["masked_logits"], val_raw["direct_logits"])}


def write_markdown(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# M8.1 Windowed FS-AVProve-Net",
        "",
        "FS-AVProve-Net is a hierarchical, predicate-factorized evidence graph verifier. It performs dependency-constrained reasoning over carrier, semantic-window, availability, synchronization, and signed-registry evidence, then admits verified only through positive-support logic plus hard cryptographic/semantic masks.",
        "",
        "## Headline",
        "",
        "```json",
        json.dumps(summary.get("headline", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Train / Val",
        "",
        "```json",
        json.dumps(summary.get("training", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Safety Interpretation",
        "",
        "- `direct_*_without_predicate_logic` reports the unconstrained state head before predicate-factorized verified logic.",
        "- `raw_*_without_safety_mask` reports predicate-constrained logits before hard gates.",
        "- `unsafe_false_verified` is the final masked router false-verified count over non-verified labels.",
        "- Counterfactual training masks claim/audio/video/sync evidence, corrupts claim validity, and replaces cross-sample video-window evidence so missing or conflicting evidence is not learned as positive support.",
        "- The signed registry and frozen M3/M4 verifier are not trained.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-packet-rows", action="append", default=["rq1_eval/artifacts_m7_audio_packet_beacon_val5k/m7_audio_packet_beacon_rows_shard*.csv"])
    parser.add_argument("--video-pointer-rows", type=Path, default=Path("rq1_eval/artifacts_m7_videoseal_short_pointer_v2_val5k/m7_videoseal_short_pointer_v2_val5k_rows.csv"))
    parser.add_argument("--m3-fused", type=Path, default=Path("rq1_eval/artifacts_m7_m6_e2e_scale160/m3_fusion/m3_fused_rows_m6_m7_enriched.csv"))
    parser.add_argument("--video-window-scores", type=Path, default=Path("rq1_eval/artifacts_m7_m6_e2e_scale160/video_window_evaluation/m2_video_window_scores.csv"))
    parser.add_argument("--audio-window-scores", type=Path, default=Path("rq1_eval/artifacts_m7_m6_e2e_scale160/audio_window_span05_hop025/segmented_evt_bch_scores.csv"))
    parser.add_argument("--claim-rows", type=Path, default=Path("rq1_eval/artifacts_m7_m6_e2e_scale160/claim_recovery/m7_short_pointer_claim_rows.csv"))
    parser.add_argument("--signed-m4-rows", type=Path, default=Path("rq1_eval/artifacts_m7_m6_e2e_scale160/signed_m4_verification/m6_signed_m4_media_realized_rows.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("rq1_eval/artifacts_m81_windowed_fs_avprove_net"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-video-windows", type=int, default=48)
    parser.add_argument("--max-audio-failed-windows", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--limit-carrier", type=int, default=12000)
    parser.add_argument("--limit-examples", type=int, default=0)
    parser.add_argument("--direct-state-loss-weight", type=float, default=0.2)
    parser.add_argument("--audio-loss-weight", type=float, default=0.2)
    parser.add_argument("--video-loss-weight", type=float, default=0.2)
    parser.add_argument("--localization-loss-weight", type=float, default=0.15)
    parser.add_argument("--predicate-loss-weight", type=float, default=0.35)
    parser.add_argument("--safety-loss-weight", type=float, default=2.0)
    parser.add_argument("--calibration-loss-weight", type=float, default=0.05)
    parser.add_argument("--counterfactual-dropout", type=float, default=0.15)
    args = parser.parse_args()
    examples, meta = build_examples(args)
    data = normalize_tokens(materialize_dataset(examples, args.max_tokens, args.train_ratio))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "m81_windowed_evidence_dataset.npz", tokens=data["tokens"], token_types=data["token_types"], token_mask=data["token_mask"], loc_labels=data["loc_labels"], state_labels=data["state_labels"], audio_labels=data["audio_labels"], video_labels=data["video_labels"], predicate_labels=data["predicate_labels"], state_masks=data["state_masks"], split=data["split"], normalization_mean=data["normalization_mean"], normalization_std=data["normalization_std"])
    write_csv(args.out_dir / "m81_windowed_evidence_dataset_index.csv", [{"row_index": i, "sample_id": data["sample_ids"][i], "source": data["sources"][i], "split": data["split"][i], "state_label": STATES[int(data["state_labels"][i])], "verified_allowed_after_mask": bool(data["state_masks"][i, STATE_TO_ID[VERIFIED]])} for i in range(len(data["state_labels"]))])
    model, training = train(data, args)
    architecture = {
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "layers": args.layers,
        "dropout": args.dropout,
        "max_tokens": args.max_tokens,
        "encoder": "hierarchical_evidence_graph",
        "token_dependency_graph": True,
        "typed_slot_graph": True,
        "decoder": "predicate-factorized positive-support logic",
    }
    torch.save({"policy": POLICY, "state_dict": model.state_dict(), "states": STATES, "token_types": TOKEN_TYPES, "predicates": PREDICATES, "feat_dim": FEAT_DIM, "normalization_mean": data["normalization_mean"].tolist(), "normalization_std": data["normalization_std"].tolist(), "architecture": architecture}, args.out_dir / "m81_windowed_fs_avprove_net.pt")
    summary = {"policy": POLICY, "headline": {**meta, "checkpoint": str(args.out_dir / "m81_windowed_fs_avprove_net.pt"), "dataset_npz": str(args.out_dir / "m81_windowed_evidence_dataset.npz"), "router": "predicate-factorized positive-support decoder + hard safety mask", "evidence_encoder": f"{args.layers}-layer hierarchical evidence graph encoder, hidden={args.hidden_dim}, heads={args.num_heads}", "predicates": PREDICATES, "counterfactual_dropout": args.counterfactual_dropout, "counterfactual_operations": ["claim mask", "audio-carrier mask", "video-carrier mask", "video-window mask", "sync mask", "claim corruption", "cross-sample video-window replacement"], "hard_gates": ["signature/source-scope/claim recovery", "frozen semantic tamper evidence"]}, "training": training, "inputs": {"audio_packet_rows": args.audio_packet_rows, "video_pointer_rows": str(args.video_pointer_rows), "m3_fused": str(args.m3_fused), "video_window_scores": str(args.video_window_scores), "audio_window_scores": str(args.audio_window_scores), "claim_rows": str(args.claim_rows), "signed_m4_rows": str(args.signed_m4_rows)}}
    write_json(args.out_dir / "m81_windowed_fs_avprove_summary.json", summary)
    write_csv(args.out_dir / "m81_val_predictions_preview.csv", training.get("val_predictions", []))
    write_markdown(args.out_dir / "M81_WINDOWED_FS_AVPROVE_NET.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
