#!/usr/bin/env python3
"""Build the M7 cross-modal carrier payload registry.

M7 trains only the watermark carrier. The semantic verifier remains frozen:
payload bits point to the signed M4 semantic/window registry, while full helper
data, Merkle roots, and signatures stay in the registry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from registry_signatures import canonical, read_jsonl, write_jsonl
from m7_payload_codec import AUDIO_BCH_V2_POLICY, bits_to_hex, encode_audio_v2_from_fragments


DEFAULT_POLICY = {
    "source_pointer_bits": 32,
    "nonce_bits": 64,
    "registry_record_bits": 64,
    "audio_commitment_bits": 32,
    "video_commitment_bits": 32,
    "sync_commitment_bits": 32,
    "ecc_parity_bits": 32,
}


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hex_fragment(value: str, n_bits: int) -> str:
    if n_bits <= 0:
        return ""
    n_hex = (n_bits + 3) // 4
    raw = value.lower()
    if not raw or any(c not in "0123456789abcdef" for c in raw):
        raw = sha256_hex(raw.encode("utf-8"))
    return raw[:n_hex]


def prf(master_key: str, *parts: object) -> str:
    h = hashlib.sha256()
    h.update(master_key.encode("utf-8"))
    for part in parts:
        h.update(b"|")
        h.update(str(part).encode("utf-8"))
    return h.hexdigest()


def first_component(record: Dict[str, Any], component_name: str) -> Dict[str, Any]:
    for component in record.get("components", []):
        if component.get("component") == component_name:
            return dict(component)
    return {}


def source_group_split(source_id: str, train_frac: float, val_frac: float) -> str:
    bucket = int(hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < train_frac:
        return "train"
    if bucket < train_frac + val_frac:
        return "val"
    return "test"


def pack_hex(fragments: Iterable[str]) -> Tuple[str, int]:
    payload_hex = "".join(fragment.lower() for fragment in fragments if fragment)
    return payload_hex, len(payload_hex) * 4


def load_signed_m4(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for envelope in read_jsonl(path):
        manifest = dict(envelope.get("manifest", {}))
        enroll_id = str(manifest.get("enroll_id", ""))
        if not enroll_id:
            continue
        records[enroll_id] = {
            "envelope_record_sha256": str(envelope.get("record_sha256", "")),
            "envelope_policy_version": str(envelope.get("policy_version", "")),
            "signature_scheme": str(envelope.get("signature_scheme", "")),
            "manifest": manifest,
        }
    return records


def bootstrap_m4_record(row: Dict[str, str], master_key: str) -> Dict[str, Any]:
    """Create unique training-only semantic roots when M4 is not materialized.

    Formal M7 carrier training can start before every large-scale clean clip has
    an explicit M4 secure-sketch record. These bootstrap roots are keyed,
    deterministic placeholders for payload composition only; they are not
    reported as M4 cryptographic/security evidence.
    """

    enroll_id = str(row["enroll_id"])
    sample_id = str(row["sample_id"])
    source_id = str(row["source_id"])
    path = str(row.get("video_path", ""))
    audio_root = prf(master_key, "formal-bootstrap-audio-root", source_id, enroll_id, path)
    video_root = prf(master_key, "formal-bootstrap-video-root", source_id, enroll_id, path)
    semantic_root = prf(master_key, "formal-bootstrap-semantic-root", source_id, sample_id, audio_root, video_root)
    fused_sha = prf(master_key, "formal-bootstrap-m3-fused", source_id, sample_id, semantic_root)
    return {
        "envelope_record_sha256": prf(master_key, "formal-bootstrap-m4-envelope", source_id, enroll_id, path),
        "envelope_policy_version": "tdsc-m7-formal-training-bootstrap-v1",
        "signature_scheme": "bootstrap-prf-no-m4-signature",
        "manifest": {
            "merkle_root": semantic_root,
            "components": [
                {
                    "component": "audio_segmented_bch",
                    "merkle_root": audio_root,
                    "status": "formal_training_bootstrap",
                },
                {
                    "component": "video_window_commitment",
                    "merkle_root": video_root,
                    "status": "formal_training_bootstrap",
                },
                {
                    "component": "m3_fused_policy",
                    "checkpoint": "tdsc-m7-formal-training-bootstrap-v1",
                    "m3_fused_sha256": fused_sha,
                    "status": "formal_training_bootstrap",
                },
            ],
        },
    }


def payload_for_record(
    row: Dict[str, str],
    m4: Dict[str, Any],
    master_key: str,
    policy: Dict[str, int],
) -> Dict[str, Any]:
    manifest = dict(m4["manifest"])
    enroll_id = str(row["enroll_id"])
    source_id = str(row["source_id"])
    sample_id = str(row["sample_id"])
    semantic_root = str(manifest.get("merkle_root", ""))
    registry_record = str(m4.get("envelope_record_sha256", ""))

    audio_component = first_component(manifest, "audio_segmented_bch")
    video_component = first_component(manifest, "video_window_commitment")
    m3_component = first_component(manifest, "m3_fused_policy")

    source_pointer = f"tdsc://source/{source_id}/enroll/{enroll_id}"
    nonce_hex = prf(master_key, "m7-nonce", enroll_id, sample_id)[:32]
    source_fragment = hex_fragment(prf(master_key, "source-pointer", source_pointer), policy["source_pointer_bits"])
    nonce_fragment = hex_fragment(nonce_hex, policy["nonce_bits"])
    registry_fragment = hex_fragment(registry_record, policy["registry_record_bits"])
    audio_fragment = hex_fragment(
        str(audio_component.get("merkle_root", "")) or prf(master_key, "missing-audio", enroll_id),
        policy["audio_commitment_bits"],
    )
    video_fragment = hex_fragment(
        str(video_component.get("merkle_root", "")) or prf(master_key, "missing-video", enroll_id),
        policy["video_commitment_bits"],
    )
    sync_material = "|".join([
        str(m3_component.get("checkpoint", "")),
        str(m3_component.get("m3_fused_sha256", "")),
        semantic_root,
    ])
    sync_fragment = hex_fragment(prf(master_key, "sync-fragment", sync_material), policy["sync_commitment_bits"])

    common = [source_fragment, nonce_fragment, registry_fragment]
    video_core, _ = pack_hex(common + [audio_fragment, sync_fragment])
    av_core, _ = pack_hex(common + [audio_fragment, video_fragment, sync_fragment])
    audio_seed = prf(master_key, "audio-modulation", source_id, nonce_hex)[:16]
    audio_bits, audio_meta = encode_audio_v2_from_fragments(
        source_fragment,
        nonce_hex,
        registry_fragment,
        audio_seed,
        master_key,
        t=16,
    )

    def with_ecc(core_hex: str, tag: str) -> Dict[str, Any]:
        ecc = hex_fragment(prf(master_key, "carrier-ecc", tag, core_hex), policy["ecc_parity_bits"])
        payload_hex, n_bits = pack_hex([core_hex, ecc])
        return {
            "payload_hex": payload_hex,
            "payload_n_bits": n_bits,
            "ecc_parity_fragment_hex": ecc,
        }

    payloads = {
        "audio": {
            "payload_hex": bits_to_hex(audio_bits),
            "payload_n_bits": len(audio_bits),
            "payload_codec": AUDIO_BCH_V2_POLICY,
            "payload_data_hex": str(audio_meta["audio_bch_data_hex"]),
            "payload_data_n_bits": int(audio_meta["audio_bch_k"]),
            "payload_bch_t": int(audio_meta["audio_bch_t"]),
            "payload_bch_n": int(audio_meta["audio_bch_n"]),
            "payload_bch_k": int(audio_meta["audio_bch_k"]),
            "payload_interleaver": str(audio_meta["audio_bch_interleaver"]),
            "carrier": "Ours-AudioCarrier",
            "cross_modal_binding": "audio carrier transports BCH-coded source pointer, nonce, and registry lookup fragment",
            "bound_fragments": {
                "source_pointer": source_fragment,
                "nonce": nonce_fragment,
                "registry_record_hash": registry_fragment,
                "policy_version_bits": str(audio_meta["audio_bch_version_bits"]),
            },
            "keyed_modulation_seed_hex": audio_seed,
        },
        "video": {
            **with_ecc(video_core, "video"),
            "carrier": "Ours-VideoCarrier",
            "cross_modal_binding": "video carrier transports audio/sync commitment fragments",
            "bound_fragments": {
                "source_pointer": source_fragment,
                "nonce": nonce_fragment,
                "registry_record_hash": registry_fragment,
                "audio_commitment": audio_fragment,
                "sync_commitment": sync_fragment,
            },
            "keyed_modulation_seed_hex": prf(master_key, "video-modulation", source_id, nonce_hex)[:16],
        },
        "av_full": {
            **with_ecc(av_core, "av-full"),
            "carrier": "Ours-AV-Full",
            "cross_modal_binding": "audio and video carriers jointly bind semantic/window commitments",
            "bound_fragments": {
                "source_pointer": source_fragment,
                "nonce": nonce_fragment,
                "registry_record_hash": registry_fragment,
                "audio_commitment": audio_fragment,
                "video_commitment": video_fragment,
                "sync_commitment": sync_fragment,
            },
            "keyed_modulation_seed_hex": prf(master_key, "av-modulation", source_id, nonce_hex)[:16],
        },
    }

    return {
        "source_pointer": source_pointer,
        "nonce_hex": nonce_hex,
        "registry_record_hash_fragment_hex": registry_fragment,
        "audio_commitment_fragment_hex": audio_fragment,
        "video_commitment_fragment_hex": video_fragment,
        "sync_commitment_fragment_hex": sync_fragment,
        "payloads": payloads,
        "m4_registry_record_sha256": registry_record,
        "m4_semantic_merkle_root": semantic_root,
        "m4_audio_merkle_root": str(audio_component.get("merkle_root", "")),
        "m4_video_window_merkle_root": str(video_component.get("merkle_root", "")),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--signed-m4-registry", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-registry", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, default=None)
    parser.add_argument("--master-key", default="tdsc-m7-cross-modal-payload-v1")
    parser.add_argument("--policy-version", default="tdsc-m7-cross-modal-payload-v1")
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-missing-m4", action="store_true")
    parser.add_argument(
        "--registry-mode",
        default="m4_bound",
        choices=["m4_bound", "formal_training_bootstrap"],
        help="Annotates whether records are bound to materialized M4 evidence or training bootstrap placeholders.",
    )
    parser.add_argument(
        "--m7-split-field",
        default="m7_split",
        help="If present in the clean manifest, use this source-disjoint split instead of recomputing it.",
    )
    args = parser.parse_args()

    clean = pd.read_csv(args.clean_manifest).fillna("")
    required = {"sample_id", "enroll_id", "source_id", "split", "video_path"}
    missing = required - set(clean.columns)
    if missing:
        raise ValueError(f"clean manifest missing columns: {sorted(missing)}")

    m4_by_enroll = load_signed_m4(args.signed_m4_registry)
    rows: List[Dict[str, str]] = []
    for _, row in clean.iterrows():
        item = {key: str(row.get(key, "")) for key in clean.columns}
        if str(item.get("population", "orig")).lower() != "orig":
            continue
        if not args.allow_missing_m4 and item["enroll_id"] not in m4_by_enroll:
            continue
        rows.append(item)
    rows.sort(key=lambda x: (x["source_id"], x["sample_id"]))
    if args.limit:
        rows = rows[: args.limit]

    policy = dict(DEFAULT_POLICY)
    registry_records: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    m4_bound_records = 0
    bootstrap_records = 0
    for row in rows:
        m4 = m4_by_enroll.get(row["enroll_id"])
        semantic_registry_status = "m4_bound"
        if m4 is None:
            if not args.allow_missing_m4:
                continue
            m4 = bootstrap_m4_record(row, args.master_key)
            semantic_registry_status = "formal_training_bootstrap"
            bootstrap_records += 1
        else:
            m4_bound_records += 1
        payload = payload_for_record(row, m4, args.master_key, policy)
        manifest_split = str(row.get(args.m7_split_field, ""))
        m7_split = manifest_split if manifest_split in {"train", "val", "test"} else source_group_split(row["source_id"], args.train_frac, args.val_frac)
        registry_record = {
            "registry_schema": "tdsc-m7-payload-registry-v1",
            "record_type": "cross_modal_carrier_payload",
            "policy_version": args.policy_version,
            "registry_mode": args.registry_mode,
            "semantic_registry_status": semantic_registry_status,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_id": row["source_id"],
            "sample_id": row["sample_id"],
            "enroll_id": row["enroll_id"],
            "original_split": row.get("split", ""),
            "m7_split": m7_split,
            "video_path": row.get("video_path", ""),
            "payload_policy": policy,
            **payload,
        }
        registry_record["registry_record_sha256"] = sha256_hex(canonical(registry_record))
        registry_records.append(registry_record)
        manifest_rows.append({
            "sample_id": row["sample_id"],
            "enroll_id": row["enroll_id"],
            "source_id": row["source_id"],
            "original_split": row.get("split", ""),
            "m7_split": m7_split,
            "population": row.get("population", "orig"),
            "cond": row.get("cond", "clean"),
            "generator": row.get("generator", "none"),
            "video_path": row.get("video_path", ""),
            "m7_payload_registry_record_sha256": registry_record["registry_record_sha256"],
            "m4_registry_record_sha256": payload["m4_registry_record_sha256"],
            "registry_mode": args.registry_mode,
            "semantic_registry_status": semantic_registry_status,
            "audio_payload_hex": payload["payloads"]["audio"]["payload_hex"],
            "audio_payload_codec": payload["payloads"]["audio"].get("payload_codec", ""),
            "audio_payload_data_hex": payload["payloads"]["audio"].get("payload_data_hex", ""),
            "audio_payload_data_n_bits": payload["payloads"]["audio"].get("payload_data_n_bits", ""),
            "audio_payload_bch_t": payload["payloads"]["audio"].get("payload_bch_t", ""),
            "audio_payload_bch_n": payload["payloads"]["audio"].get("payload_bch_n", ""),
            "audio_payload_bch_k": payload["payloads"]["audio"].get("payload_bch_k", ""),
            "video_payload_hex": payload["payloads"]["video"]["payload_hex"],
            "av_full_payload_hex": payload["payloads"]["av_full"]["payload_hex"],
            "audio_payload_n_bits": payload["payloads"]["audio"]["payload_n_bits"],
            "video_payload_n_bits": payload["payloads"]["video"]["payload_n_bits"],
            "av_full_payload_n_bits": payload["payloads"]["av_full"]["payload_n_bits"],
        })

    write_jsonl(args.out_registry, registry_records)
    write_csv(args.out_manifest, manifest_rows)
    source_splits: Dict[str, str] = {}
    for row in manifest_rows:
        source_splits.setdefault(str(row["source_id"]), str(row["m7_split"]))
    split_counts = pd.Series([row["m7_split"] for row in manifest_rows]).value_counts().to_dict() if manifest_rows else {}
    summary = {
        "clean_manifest": str(args.clean_manifest),
        "signed_m4_registry": str(args.signed_m4_registry),
        "out_manifest": str(args.out_manifest),
        "out_registry": str(args.out_registry),
        "records": len(registry_records),
        "sources": len(source_splits),
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "m4_bound_records": m4_bound_records,
        "formal_training_bootstrap_records": bootstrap_records,
        "registry_mode": args.registry_mode,
        "policy_version": args.policy_version,
        "payload_policy": policy,
        "note": "M7 payloads carry registry pointers/fragments only; full signatures and helper data remain in signed registries. Bootstrap records are for carrier training and are not M4 security evidence.",
    }
    out_summary = args.out_summary or args.out_registry.with_suffix(".summary.json")
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
