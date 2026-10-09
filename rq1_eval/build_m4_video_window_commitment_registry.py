#!/usr/bin/env python3
"""Materialize video-window helper/digest/nonce/Merkle records for M4.5.

The current M2 video-window gate uses 255-bit keyed LSH codes and per-channel
Hamming thresholds. `evt_video` at t=37 can be represented by a real BCH(255)
code-offset commitment. `id_video` at t≈141 cannot be represented by BCH(255)
with positive dimension, so this prototype uses the project's RadiusECC fallback
and marks those segments as `secure=false` in the registry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from fuzzy import FuzzyCommitment, RadiusECC, make_bch_fe
from quantize import LSHQuantizer
from registry_signatures import canonical

CHANNELS = ["id_video", "evt_video"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def merkle_root(leaves: List[bytes]) -> str:
    if not leaves:
        return hashlib.sha256(b"tdsc-empty-video-window-registry").hexdigest()
    nodes = [hashlib.sha256(b"\x00" + leaf).digest() for leaf in leaves]
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [
            hashlib.sha256(b"\x01" + nodes[i] + nodes[i + 1]).digest()
            for i in range(0, len(nodes), 2)
        ]
    return nodes[0].hex()


def nonce_and_seed(master_key: bytes, enroll_id: str, channel: str, index: int) -> Tuple[bytes, int]:
    h = hashlib.sha256()
    h.update(master_key)
    h.update(b"|video-window|")
    h.update(enroll_id.encode("utf-8"))
    h.update(b"|")
    h.update(channel.encode("utf-8"))
    h.update(b"|")
    h.update(str(index).encode("ascii"))
    digest = h.digest()
    return digest[:16], int.from_bytes(digest[16:24], "big")


def threshold_from_summaries(paths: List[Path]) -> Dict[str, int]:
    thresholds: Dict[str, int] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for channel, value in data.get("thresholds_bits", {}).items():
            thresholds[channel] = max(thresholds.get(channel, 0), int(value))
    if not thresholds:
        thresholds = {"id_video": 141, "evt_video": 37}
    for channel in CHANNELS:
        if channel not in thresholds:
            raise ValueError(f"missing threshold for {channel}")
    return thresholds


def choose_fe(channel: str, threshold: int, n_bits: int):
    if channel == "evt_video" and n_bits == 255 and threshold <= 80:
        return make_bch_fe(8, threshold, domain=b"tdsc-video-window-v1"), {
            "ecc": "BCH",
            "m": 8,
            "secure": True,
            "reason": "BCH(255) has positive message dimension for this threshold",
        }
    return FuzzyCommitment(RadiusECC(n_bits, threshold), domain=b"tdsc-video-window-v1"), {
        "ecc": "RadiusECC",
        "m": None,
        "secure": False,
        "reason": "prototype fallback: threshold is not feasible for BCH at this n_bits",
    }


def select_orig_rows(manifests: List[Path]) -> List[Dict[str, str]]:
    rows_by_enroll: Dict[str, Dict[str, str]] = {}
    for manifest in manifests:
        for row in read_csv(manifest):
            if str(row.get("population", "")).lower() != "orig":
                continue
            enroll_id = str(row.get("enroll_id", ""))
            path = row.get("window_feature_path", "")
            if not enroll_id or not path or not Path(path).exists():
                continue
            rows_by_enroll.setdefault(enroll_id, {**row, "feature_manifest": str(manifest)})
    return list(rows_by_enroll.values())


def build_record(
    row: Dict[str, str],
    thresholds: Dict[str, int],
    n_bits: int,
    master_key: bytes,
) -> Dict[str, object]:
    enroll_id = str(row["enroll_id"])
    data = load_npz(row["window_feature_path"])
    dims = {channel: int(data[channel].shape[1]) for channel in CHANNELS}
    quantizers = {
        channel: LSHQuantizer(dims[channel], n_bits, key=master_key, tag=channel)
        for channel in CHANNELS
    }
    leaves: List[bytes] = []
    segments: List[Dict[str, object]] = []
    starts = data["starts_sec"].reshape(-1)
    for index, start in enumerate(starts):
        for channel in CHANNELS:
            available_key = f"{channel}_available"
            available = bool(data[available_key][index]) if available_key in data else True
            segment: Dict[str, object] = {
                "index": int(index),
                "start_sec": float(start),
                "channel": channel,
                "available": available,
                "threshold_bits": int(thresholds[channel]),
                "feature_dim": dims[channel],
            }
            if available:
                fe, ecc_meta = choose_fe(channel, int(thresholds[channel]), n_bits)
                code = quantizers[channel].encode(data[channel][index])
                nonce, seed = nonce_and_seed(master_key, enroll_id, channel, index)
                _, enrollment = fe.gen(code, nonce=nonce, rng=np.random.default_rng(seed))
                segment.update(ecc_meta)
                segment.update(enrollment.to_public())
            else:
                segment.update({"ecc": "unavailable", "secure": False, "reason": "window feature unavailable"})
            segment["segment_sha256"] = hashlib.sha256(canonical(segment)).hexdigest()
            segments.append(segment)
            leaves.append(canonical(segment))
    record = {
        "registry_schema": "tdsc-video-window-commitment-v1",
        "record_type": "video_window_commitment_registry",
        "source_id": str(row.get("source_id", "")),
        "sample_id": str(row.get("sample_id", "")),
        "enroll_id": enroll_id,
        "split": str(row.get("split", "")),
        "feature_manifest": str(row.get("feature_manifest", "")),
        "window_feature_path": str(row.get("window_feature_path", "")),
        "window_feature_sha256": sha256_file(Path(row.get("window_feature_path", ""))),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "policy_version": "tdsc-video-window-commitment-v1",
            "quantizer_kind": "lsh",
            "quantizer_master_key_tag": "rq1-m2-video-window-stage0-compatible",
            "n_bits": n_bits,
            "thresholds_bits": thresholds,
            "window_span_sec": 1.0,
            "window_hop_sec": 0.5,
        },
        "window_count": int(len(starts)),
        "segment_count": len(segments),
        "segments": segments,
        "merkle_root": merkle_root(leaves),
    }
    record["registry_record_sha256"] = hashlib.sha256(canonical(record)).hexdigest()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-feature-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--threshold-summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--out-registry", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, default=None)
    parser.add_argument("--n-bits", type=int, default=255)
    parser.add_argument("--master-key", default="rq1-m2-video-window-stage0")
    args = parser.parse_args()

    thresholds = threshold_from_summaries(args.threshold_summaries)
    rows = select_orig_rows(args.window_feature_manifests)
    records = [build_record(row, thresholds, args.n_bits, args.master_key.encode("utf-8")) for row in rows]
    write_jsonl(args.out_registry, records)
    secure_segments = 0
    total_segments = 0
    by_ecc: Dict[str, int] = {}
    for record in records:
        for segment in record["segments"]:
            total_segments += 1
            secure_segments += int(bool(segment.get("secure")))
            by_ecc[str(segment.get("ecc", ""))] = by_ecc.get(str(segment.get("ecc", "")), 0) + 1
    summary = {
        "registry": str(args.out_registry),
        "records": len(records),
        "thresholds_bits": thresholds,
        "n_bits": args.n_bits,
        "segments": total_segments,
        "secure_segments": secure_segments,
        "insecure_or_fallback_segments": total_segments - secure_segments,
        "segments_by_ecc": by_ecc,
        "window_feature_manifests": [str(p) for p in args.window_feature_manifests],
        "threshold_summaries": [str(p) for p in args.threshold_summaries],
        "scope_note": (
            "evt_video uses real BCH when feasible; id_video uses RadiusECC fallback "
            "because the frozen M2 threshold is too large for BCH(255)."
        ),
    }
    out_summary = args.out_summary or args.out_registry.with_suffix(".summary.json")
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
