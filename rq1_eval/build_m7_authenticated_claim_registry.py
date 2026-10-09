#!/usr/bin/env python3
"""Build the signed-registry input for authenticated, versioned claim locators."""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from registry_signatures import canonical, load_public, read_jsonl, unsigned, write_jsonl


LOCATOR_POLICY = "tdsc-m7-authenticated-claim-locator-v2"
FAMILY_FIELDS = (
    "source_id",
    "enroll_id",
    "sample_id",
    "source_pointer",
    "semantic_registry_status",
    "m4_registry_record_sha256",
    "m4_audio_merkle_root",
    "m4_semantic_merkle_root",
    "m4_video_window_merkle_root",
    "audio_commitment_fragment_hex",
    "video_commitment_fragment_hex",
    "sync_commitment_fragment_hex",
    "policy_version",
)


def verify_envelope(public: Any, envelope: Mapping[str, Any]) -> None:
    payload = canonical(unsigned(dict(envelope)))
    if hashlib.sha256(payload).hexdigest() != str(envelope.get("record_sha256", "")):
        raise ValueError("source envelope digest mismatch")
    signature = base64.b64decode(
        str(envelope.get("signature_ed25519_b64", "")), validate=True
    )
    public.verify(signature, payload)


def family_digest(manifest: Mapping[str, Any]) -> str:
    material = {field: manifest.get(field) for field in FAMILY_FIELDS}
    return hashlib.sha256(canonical(material)).hexdigest()


def locator_hex(
    family_sha256: str,
    version: int,
    previous_record_sha256: str,
    salt: int,
) -> str:
    material = "|".join(
        [LOCATOR_POLICY, family_sha256, str(version), previous_record_sha256, str(salt)]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:8]


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def load_parts(parts: Sequence[Sequence[str]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_hashes = set()
    for signed_text, public_text in parts:
        signed_path = Path(signed_text)
        public_path = Path(public_text)
        public = load_public(public_path)
        for envelope in read_jsonl(signed_path):
            verify_envelope(public, envelope)
            manifest = dict(envelope.get("manifest", {}))
            record_hash = str(manifest.get("registry_record_sha256", ""))
            if not record_hash:
                raise ValueError(f"missing registry_record_sha256 in {signed_path}")
            if record_hash in seen_hashes:
                continue
            seen_hashes.add(record_hash)
            records.append(
                {
                    "manifest": manifest,
                    "previous_registry_record_sha256": record_hash,
                    "previous_envelope_record_sha256": str(
                        envelope.get("record_sha256", "")
                    ),
                    "source_signed_registry": str(signed_path),
                }
            )
    return records


def build_authenticated_registry(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    families: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in records:
        item["claim_family_sha256"] = family_digest(item["manifest"])
        families[item["claim_family_sha256"]].append(item)
    for members in families.values():
        members.sort(
            key=lambda item: (
                str(item["manifest"].get("created_at", "")),
                item["previous_registry_record_sha256"],
            )
        )
        for version, item in enumerate(members, start=1):
            item["claim_version"] = version

    used_audio32 = set()
    used_video24 = set()
    registry: List[Dict[str, Any]] = []
    mapping: List[Dict[str, Any]] = []
    ordered = sorted(
        records,
        key=lambda item: (
            str(item["manifest"].get("source_id", "")),
            str(item["manifest"].get("enroll_id", "")),
            int(item["claim_version"]),
            item["previous_registry_record_sha256"],
        ),
    )
    for item in ordered:
        salt = 0
        while True:
            locator = locator_hex(
                item["claim_family_sha256"],
                int(item["claim_version"]),
                item["previous_registry_record_sha256"],
                salt,
            )
            if locator not in used_audio32 and locator[:6] not in used_video24:
                break
            salt += 1
        used_audio32.add(locator)
        used_video24.add(locator[:6])

        manifest = copy.deepcopy(item["manifest"])
        manifest.update(
            {
                "registry_schema": "tdsc-m7-payload-registry-v2",
                "claim_locator_policy": LOCATOR_POLICY,
                "claim_family_sha256": item["claim_family_sha256"],
                "claim_version": int(item["claim_version"]),
                "claim_locator_salt": salt,
                "authenticated_claim_locator_hex": locator,
                "authenticated_claim_locator_bits": 32,
                "video_claim_locator_bits": 24,
                "previous_registry_record_sha256": item[
                    "previous_registry_record_sha256"
                ],
                "previous_envelope_record_sha256": item[
                    "previous_envelope_record_sha256"
                ],
            }
        )
        payload_policy = dict(manifest.get("payload_policy", {}))
        payload_policy.update(
            {
                "claim_locator_policy": LOCATOR_POLICY,
                "authenticated_claim_locator_bits": 32,
                "video_claim_locator_bits": 24,
            }
        )
        manifest["payload_policy"] = payload_policy
        for payload_name in ("audio", "video", "av_full"):
            payload = dict(manifest.get("payloads", {}).get(payload_name, {}))
            bound = dict(payload.get("bound_fragments", {}))
            bound.update(
                {
                    "authenticated_claim_locator": locator,
                    "claim_version": str(item["claim_version"]),
                }
            )
            payload["bound_fragments"] = bound
            manifest.setdefault("payloads", {})[payload_name] = payload

        manifest.pop("registry_record_sha256", None)
        new_hash = hashlib.sha256(canonical(manifest)).hexdigest()
        manifest["registry_record_sha256"] = new_hash
        registry.append(manifest)
        mapping.append(
            {
                "sample_id": manifest.get("sample_id", ""),
                "source_id": manifest.get("source_id", ""),
                "enroll_id": manifest.get("enroll_id", ""),
                "claim_family_sha256": item["claim_family_sha256"],
                "claim_version": item["claim_version"],
                "claim_locator_salt": salt,
                "authenticated_claim_locator_hex": locator,
                "video_claim_locator_hex": locator[:6],
                "previous_registry_record_sha256": item[
                    "previous_registry_record_sha256"
                ],
                "registry_record_sha256": new_hash,
            }
        )
    return registry, mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part",
        nargs=2,
        action="append",
        metavar=("SIGNED_REGISTRY", "PUBLIC_KEY"),
        required=True,
    )
    parser.add_argument("--out-registry", type=Path, required=True)
    parser.add_argument("--out-mapping", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source_records = load_parts(args.part)
    if args.limit:
        source_records = source_records[:args.limit]
    registry, mapping = build_authenticated_registry(source_records)
    write_jsonl(args.out_registry, registry)
    write_csv(args.out_mapping, mapping)
    family_counts: Dict[str, int] = defaultdict(int)
    for row in mapping:
        family_counts[str(row["claim_family_sha256"])] += 1
    summary = {
        "policy": LOCATOR_POLICY,
        "records": len(mapping),
        "claim_families": len(family_counts),
        "versioned_families": sum(count > 1 for count in family_counts.values()),
        "max_versions_per_family": max(family_counts.values(), default=0),
        "audio32_unique": len({row["authenticated_claim_locator_hex"] for row in mapping}),
        "video24_unique": len({row["video_claim_locator_hex"] for row in mapping}),
        "salted_collision_resolutions": sum(
            int(row["claim_locator_salt"]) > 0 for row in mapping
        ),
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
