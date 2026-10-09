#!/usr/bin/env python3
"""Evaluate segmented EVT-audio BCH reconstruction and fail-safe policies.

Each enrolled audio window is quantized independently. A query window is
recoverable iff its keyed-LSH code is within the BCH correction radius of the
nearest registered window code. The resulting per-window recovery mask drives
policy-level provenance states:

    verified             recovery ratio >= policy threshold
    localized-tamper     failed windows form a minority and can be localized
    content-unverified   too many windows fail to verify content integrity

The default BCH candidate comes from EVT_AUDIO_V2_SPLICE200_RESULT.md:

    BCH(511, 139, t=54)

Use --real-bch-smoke to run actual code-offset FuzzyCommitment.gen/verify on a
small subset and confirm that the radius predicate agrees with the real codec.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from fuzzy import FuzzyCommitment, make_bch_fe
from quantize import LSHQuantizer


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def relocate(path: str, mappings: List[Tuple[str, str]]) -> Path:
    for old, new in mappings:
        if path.startswith(old):
            path = new + path[len(old):]
            break
    return Path(path)


def load_npy(row: Dict[str, str], col: str, mappings: List[Tuple[str, str]]) -> np.ndarray:
    return np.load(relocate(row[col], mappings))


def merkle_root(leaves: List[bytes]) -> str:
    """Return a deterministic SHA-256 Merkle root for ordered segment leaves."""
    if not leaves:
        return hashlib.sha256(b"rq1-empty-merkle").hexdigest()
    nodes = [hashlib.sha256(b"\x00" + leaf).digest() for leaf in leaves]
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [
            hashlib.sha256(b"\x01" + nodes[i] + nodes[i + 1]).digest()
            for i in range(0, len(nodes), 2)
        ]
    return nodes[0].hex()


def window_codes(
    row: Dict[str, str],
    mappings: List[Tuple[str, str]],
    quantizer: LSHQuantizer,
) -> Tuple[np.ndarray, np.ndarray]:
    windows = load_npy(row, "evt_audio_windows_path", mappings)
    starts = load_npy(row, "evt_audio_window_starts_path", mappings).reshape(-1)
    return quantizer.encode_batch(windows), starts


def contiguous_failed_run(mask: np.ndarray) -> int:
    longest = current = 0
    for ok in mask:
        current = 0 if ok else current + 1
        longest = max(longest, current)
    return longest


def classify(mask: np.ndarray, policy: float, tamper_run_threshold: int) -> str:
    ratio = float(np.mean(mask)) if len(mask) else 0.0
    if ratio >= policy and contiguous_failed_run(mask) < tamper_run_threshold:
        return "verified"
    if ratio >= 0.5:
        return "localized-tamper"
    return "content-unverified"


def compare(
    row: Dict[str, str],
    orig: Dict[str, str],
    mappings: List[Tuple[str, str]],
    quantizer: LSHQuantizer,
    t: int,
    policies: List[float],
    tamper_run_threshold: int,
    enrolled_merkle_root: str,
) -> Dict[str, object]:
    ref_codes, ref_starts = window_codes(orig, mappings, quantizer)
    qry_codes, qry_starts = window_codes(row, mappings, quantizer)
    ham_bits = []
    ref_index = []
    for code, start in zip(qry_codes, qry_starts):
        idx = int(np.argmin(np.abs(ref_starts - start)))
        ref_index.append(idx)
        ham_bits.append(int(np.count_nonzero(code != ref_codes[idx])))
    ham_bits = np.asarray(ham_bits, dtype=np.int32)
    recovered = ham_bits <= t
    failed_starts = [float(s) for s, ok in zip(qry_starts, recovered) if not ok]

    rec: Dict[str, object] = {
        "sample_id": row["sample_id"],
        "enroll_id": row["enroll_id"],
        "source_id": row["source_id"],
        "population": row["population"],
        "cond": row["cond"],
        "generator": row["generator"],
        "split": row["split"],
        "attack_scope": row.get("attack_scope", ""),
        "target_start_sec": row.get("target_start_sec", ""),
        "target_end_sec": row.get("target_end_sec", ""),
        "attack_variant": row.get("attack_variant", ""),
        "splice_duration_sec": row.get("splice_duration_sec", ""),
        "splice_position": row.get("splice_position", ""),
        "codec_bandwidth": row.get("codec_bandwidth", ""),
        "enhancement_model": row.get("enhancement_model", ""),
        "attack_language": row.get("attack_language", ""),
        "window_count": len(recovered),
        "recovered_windows": int(recovered.sum()),
        "failed_windows": int((~recovered).sum()),
        "recovery_ratio": float(np.mean(recovered)) if len(recovered) else 0.0,
        "longest_failed_run": contiguous_failed_run(recovered),
        "max_ham_bits": int(ham_bits.max()) if len(ham_bits) else 0,
        "q90_ham_bits": float(np.quantile(ham_bits, 0.9)) if len(ham_bits) else 0.0,
        "failed_window_starts_json": json.dumps(failed_starts, separators=(",", ":")),
        "enrolled_merkle_root": enrolled_merkle_root,
    }
    for policy in policies:
        tag = f"p{int(round(policy * 100)):03d}"
        rec[f"state_{tag}"] = classify(recovered, policy, tamper_run_threshold)
    return rec


def canonical_json_bytes(record: Dict[str, object]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


def enrollment_seed(master_key: bytes, enroll_id: str, index: int) -> Tuple[bytes, int]:
    material = master_key + b"|rq1-segment|" + enroll_id.encode() + b"|" + str(index).encode()
    digest = hashlib.sha256(material).digest()
    return digest[:16], int.from_bytes(digest[16:24], "big")


def build_public_registry(
    orig: Dict[str, Dict[str, str]],
    mappings: List[Tuple[str, str]],
    quantizer: LSHQuantizer,
    fe: FuzzyCommitment,
    master_key: bytes,
) -> Dict[str, Dict[str, object]]:
    """Materialize helper-data records and authenticate them with Merkle roots."""
    registry = {}
    for enroll_id, row in orig.items():
        codes, starts = window_codes(row, mappings, quantizer)
        segments = []
        for index, (code, start) in enumerate(zip(codes, starts)):
            nonce, seed = enrollment_seed(master_key, enroll_id, index)
            _, enr = fe.gen(code, nonce=nonce, rng=np.random.default_rng(seed))
            segment = {
                "index": index,
                "start_sec": float(start),
                **enr.to_public(),
            }
            segments.append(segment)
        root = merkle_root([canonical_json_bytes(segment) for segment in segments])
        registry[enroll_id] = {
            "enroll_id": enroll_id,
            "source_id": row["source_id"],
            "sample_id": row["sample_id"],
            "merkle_root": root,
            "segments": segments,
        }
    return registry


def summarize(rows: List[Dict[str, object]], policies: List[float]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["split"]), str(row["population"]), str(row["cond"]),
            str(row["generator"]), str(row.get("attack_variant", "")),
        )
        groups.setdefault(key, []).append(row)
    out = []
    for key, group in sorted(groups.items()):
        rec: Dict[str, object] = {
            "split": key[0], "population": key[1], "cond": key[2], "generator": key[3],
            "attack_variant": key[4],
            "n": len(group),
            "recovery_ratio_mean": float(np.mean([r["recovery_ratio"] for r in group])),
            "recovery_ratio_p10": float(np.quantile([r["recovery_ratio"] for r in group], 0.1)),
            "failed_windows_mean": float(np.mean([r["failed_windows"] for r in group])),
            "longest_failed_run_mean": float(np.mean([r["longest_failed_run"] for r in group])),
        }
        for policy in policies:
            tag = f"p{int(round(policy * 100)):03d}"
            states = [str(r[f"state_{tag}"]) for r in group]
            for state in ("verified", "localized-tamper", "content-unverified"):
                rec[f"{tag}_{state}_rate"] = states.count(state) / len(states)
        out.append(rec)
    return out


def real_bch_smoke(
    rows: List[Dict[str, str]],
    orig: Dict[str, str],
    mappings: List[Tuple[str, str]],
    quantizer: LSHQuantizer,
    fe: FuzzyCommitment,
    limit: int,
) -> Dict[str, int]:
    rng = np.random.default_rng(0)
    checked = disagreements = 0
    for row in rows:
        if row["population"] == "orig" or row["enroll_id"] not in orig:
            continue
        ref_codes, ref_starts = window_codes(orig[row["enroll_id"]], mappings, quantizer)
        qry_codes, qry_starts = window_codes(row, mappings, quantizer)
        for code, start in zip(qry_codes, qry_starts):
            idx = int(np.argmin(np.abs(ref_starts - start)))
            ref = ref_codes[idx]
            _, enr = fe.gen(ref, rng=rng)
            actual = fe.verify(code, enr)
            ideal = int(np.count_nonzero(code != ref)) <= fe.t
            checked += 1
            disagreements += int(actual != ideal)
            if checked >= limit:
                print(f"[segmented-bch] real BCH smoke checked={checked} disagreements={disagreements}")
                if disagreements:
                    raise RuntimeError("real BCH verification disagrees with radius predicate")
                return {"checked": checked, "disagreements": disagreements}
    print(f"[segmented-bch] real BCH smoke checked={checked} disagreements={disagreements}")
    if disagreements:
        raise RuntimeError("real BCH verification disagrees with radius predicate")
    return {"checked": checked, "disagreements": disagreements}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-bits", type=int, default=511)
    ap.add_argument("--bch-m", type=int, default=9)
    ap.add_argument("--bch-t", type=int, default=54)
    ap.add_argument("--master-key", default="rq1-stage0-real")
    ap.add_argument("--policies", type=float, nargs="+", default=[1.0, 0.95, 0.90])
    ap.add_argument("--tamper-run-threshold", type=int, default=2,
                    help="contiguous failed windows that force localized-tamper")
    ap.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    ap.add_argument("--drop-orphans", action="store_true")
    ap.add_argument("--real-bch-smoke", type=int, default=0,
                    help="run real code-offset verify on this many windows")
    ap.add_argument("--materialize-registry", action="store_true",
                    help="write public helper-data segment records and Merkle roots")
    args = ap.parse_args()

    fe = make_bch_fe(args.bch_m, args.bch_t, domain=b"rq1-evt-audio-v2-segment")
    if fe.n != args.n_bits:
        raise ValueError(f"BCH(m={args.bch_m}) has n={fe.n}, expected --n-bits={args.n_bits}")
    mappings = []
    for raw in args.path_map:
        if "=" not in raw:
            ap.error(f"--path-map must have OLD=NEW form: {raw!r}")
        mappings.append(tuple(raw.split("=", 1)))
    rows = read_csv(args.manifest)
    orig_rows = [r for r in rows if r["population"] == "orig"]
    orig = {r["enroll_id"]: r for r in orig_rows}
    if len(orig) != len(orig_rows):
        raise ValueError("duplicate enrolled originals")

    first = next(iter(orig.values()))
    in_dim = int(load_npy(first, "evt_audio_windows_path", mappings).shape[1])
    quantizer = LSHQuantizer(in_dim, args.n_bits, key=args.master_key.encode(), tag="evt_audio_v2")
    if args.tamper_run_threshold < 1:
        raise ValueError("--tamper-run-threshold must be >= 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    registry = {}
    if args.materialize_registry:
        registry = build_public_registry(orig, mappings, quantizer, fe, args.master_key.encode())
        with (args.out_dir / "segmented_evt_bch_registry.jsonl").open("w", encoding="utf-8") as f:
            for enroll_id in sorted(registry):
                f.write(json.dumps(registry[enroll_id], sort_keys=True) + "\n")
        print(f"[segmented-bch] registry enrollments={len(registry)}")

    results = []
    for row in rows:
        if row["population"] == "orig":
            continue
        if row["enroll_id"] not in orig:
            if args.drop_orphans:
                continue
            raise ValueError(f"row {row['sample_id']!r} references missing original")
        enrolled_root = registry.get(row["enroll_id"], {}).get("merkle_root", "")
        results.append(compare(
            row, orig[row["enroll_id"]], mappings, quantizer, fe.t, args.policies,
            args.tamper_run_threshold, str(enrolled_root),
        ))
    write_csv(args.out_dir / "segmented_evt_bch_scores.csv", results)
    write_csv(args.out_dir / "segmented_evt_bch_summary.csv", summarize(results, args.policies))
    config = {
        "n": fe.n, "k": fe.ecc.k, "t": fe.t,
        "policies": args.policies,
        "tamper_run_threshold": args.tamper_run_threshold,
        "registry_mode": ("public-helper-records" if args.materialize_registry else "not-materialized"),
        "note": "radius-predicate dataset scan; real BCH smoke checks codec plumbing",
    }
    if args.real_bch_smoke:
        config["real_bch_smoke"] = real_bch_smoke(
            rows, orig, mappings, quantizer, fe, args.real_bch_smoke
        )
    (args.out_dir / "segmented_evt_bch_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    print(f"[segmented-bch] rows={len(results)} BCH({fe.n},{fe.ecc.k},t={fe.t}) -> {args.out_dir}")


if __name__ == "__main__":
    main()
