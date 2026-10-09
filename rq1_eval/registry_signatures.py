#!/usr/bin/env python3
"""Generate, sign, and verify Ed25519 registry envelopes.

This utility signs the helper-data registry JSONL emitted by
evaluate_segmented_evt_bch.py --materialize-registry. It requires the
`cryptography` package and never writes raw semantic embeddings.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


SIGNATURE_KEYS = {"signature_scheme", "signature_ed25519_b64", "record_sha256"}


def crypto():
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except ImportError as exc:
        raise SystemExit("registry_signatures.py requires the `cryptography` package") from exc
    return serialization, Ed25519PrivateKey, Ed25519PublicKey


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return records


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def unsigned(record: Dict) -> Dict:
    return {key: value for key, value in record.items() if key not in SIGNATURE_KEYS}


def keygen(args: argparse.Namespace) -> None:
    serialization, Ed25519PrivateKey, _ = crypto()
    private = Ed25519PrivateKey.generate()
    public = private.public_key()
    args.private_key.parent.mkdir(parents=True, exist_ok=True)
    args.public_key.parent.mkdir(parents=True, exist_ok=True)
    args.private_key.write_bytes(
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    args.public_key.write_bytes(
        public.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    print(f"[registry-signatures] private key -> {args.private_key}")
    print(f"[registry-signatures] public key -> {args.public_key}")


def load_private(path: Path):
    serialization, _, _ = crypto()
    return serialization.load_pem_private_key(path.read_bytes(), password=None)


def load_public(path: Path):
    serialization, _, _ = crypto()
    return serialization.load_pem_public_key(path.read_bytes())


def sign(args: argparse.Namespace) -> None:
    private = load_private(args.private_key)
    issued_at = args.issued_at or datetime.now(timezone.utc).isoformat()
    signed = []
    for record in read_jsonl(args.registry):
        envelope = {
            "manifest": record,
            "policy_version": args.policy_version,
            "issued_at": issued_at,
        }
        payload = canonical(envelope)
        envelope["record_sha256"] = hashlib.sha256(payload).hexdigest()
        envelope["signature_scheme"] = "Ed25519"
        envelope["signature_ed25519_b64"] = base64.b64encode(private.sign(payload)).decode("ascii")
        signed.append(envelope)
    write_jsonl(args.out, signed)
    print(f"[registry-signatures] signed={len(signed)} -> {args.out}")


def verify(args: argparse.Namespace) -> None:
    public = load_public(args.public_key)
    records = read_jsonl(args.signed_registry)
    failures = []
    for index, record in enumerate(records):
        payload_record = unsigned(record)
        payload = canonical(payload_record)
        expected_hash = hashlib.sha256(payload).hexdigest()
        try:
            if expected_hash != record.get("record_sha256"):
                raise ValueError("SHA-256 digest mismatch")
            signature = base64.b64decode(record["signature_ed25519_b64"], validate=True)
            public.verify(signature, payload)
        except Exception as exc:
            failures.append({"index": index, "error": str(exc)})
    print(f"[registry-signatures] checked={len(records)} failures={len(failures)}")
    if failures:
        print(json.dumps(failures[:10], indent=2))
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    key_parser = sub.add_parser("keygen")
    key_parser.add_argument("--private-key", type=Path, required=True)
    key_parser.add_argument("--public-key", type=Path, required=True)
    key_parser.set_defaults(func=keygen)
    sign_parser = sub.add_parser("sign")
    sign_parser.add_argument("--registry", type=Path, required=True)
    sign_parser.add_argument("--private-key", type=Path, required=True)
    sign_parser.add_argument("--out", type=Path, required=True)
    sign_parser.add_argument("--policy-version", default="rq1-stage0-v1")
    sign_parser.add_argument("--issued-at", default="")
    sign_parser.set_defaults(func=sign)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--signed-registry", type=Path, required=True)
    verify_parser.add_argument("--public-key", type=Path, required=True)
    verify_parser.set_defaults(func=verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
