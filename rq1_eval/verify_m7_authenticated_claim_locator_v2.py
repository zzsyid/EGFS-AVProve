#!/usr/bin/env python3
"""Recover and verify M7 v2 authenticated claim locators without an oracle."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch

from authenticated_locator_carrier import (
    lookup_candidates,
    neighbor_masks,
    read_csv,
    recover_audio_claim,
    recover_video_claim,
    write_csv,
)
from build_m7_authenticated_claim_registry import (
    LOCATOR_POLICY,
    family_digest,
    locator_hex as derive_locator_hex,
)
from registry_signatures import canonical, load_public, read_jsonl, unsigned


def verify_envelope(public: Any, envelope: Mapping[str, Any]) -> Tuple[bool, str]:
    try:
        payload = canonical(unsigned(dict(envelope)))
        digest = hashlib.sha256(payload).hexdigest()
        if digest != str(envelope.get("record_sha256", "")):
            return False, "envelope_digest_mismatch"
        signature = base64.b64decode(
            str(envelope.get("signature_ed25519_b64", "")), validate=True
        )
        public.verify(signature, payload)
        return True, "signature_ok"
    except Exception as exc:
        return False, f"signature_exception:{type(exc).__name__}"


def authenticated_locator(manifest: Mapping[str, Any]) -> Tuple[str, bool, str]:
    locator = str(manifest.get("authenticated_claim_locator_hex", "")).lower()
    if len(locator) != 8 or any(char not in "0123456789abcdef" for char in locator):
        return "", False, "invalid_locator_format"
    if str(manifest.get("claim_locator_policy", "")) != LOCATOR_POLICY:
        return locator, False, "locator_policy_mismatch"
    family = str(manifest.get("claim_family_sha256", ""))
    if family != family_digest(manifest):
        return locator, False, "claim_family_digest_mismatch"
    expected = derive_locator_hex(
        family,
        int(manifest.get("claim_version", 0)),
        str(manifest.get("previous_registry_record_sha256", "")),
        int(manifest.get("claim_locator_salt", 0)),
    )
    if locator != expected:
        return locator, False, "locator_derivation_mismatch"
    return locator, True, "locator_auth_ok"


def load_registry(
    signed_registry: Path,
    public_key: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[int, List[str]]]:
    public = load_public(public_key)
    by_hash: Dict[str, Dict[str, Any]] = {}
    audio_index: Dict[str, List[str]] = defaultdict(list)
    video_index: Dict[int, List[str]] = defaultdict(list)
    for envelope in read_jsonl(signed_registry):
        manifest = dict(envelope.get("manifest", {}))
        record_hash = str(manifest.get("registry_record_sha256", ""))
        if not record_hash:
            continue
        signature_ok, signature_reason = verify_envelope(public, envelope)
        locator, locator_ok, locator_reason = authenticated_locator(manifest)
        manifest["_signature_ok"] = signature_ok
        manifest["_signature_reason"] = signature_reason
        manifest["_locator_auth_ok"] = locator_ok
        manifest["_locator_auth_reason"] = locator_reason
        by_hash[record_hash] = manifest
        if signature_ok and locator_ok:
            audio_index[locator].append(record_hash)
            video_index[int(locator[:6], 16)].append(record_hash)
    for index in (audio_index, video_index):
        for key in index:
            index[key].sort()
    return by_hash, dict(audio_index), dict(video_index)


def resolve_claim(
    audio: Mapping[str, Any],
    video: Mapping[str, Any],
    by_hash: Mapping[str, Dict[str, Any]],
    audio_index: Mapping[str, List[str]],
    video_index: Mapping[int, List[str]],
    video_masks: List[Tuple[int, int]],
) -> Dict[str, Any]:
    audio_tag = str(audio.get("audio_recovered_tag_hex", "")).lower()
    audio_candidates = (
        list(audio_index.get(audio_tag, []))
        if bool(audio.get("audio_tag_recovered")) and audio_tag
        else []
    )
    video_pairs: List[Tuple[int, str]] = []
    if bool(video.get("video_pointer_recovered")):
        value = int(video.get("video_recovered_pointer_value", -1))
        if value >= 0:
            video_pairs = lookup_candidates(value, dict(video_index), video_masks)
    video_distance: Dict[str, int] = {}
    for distance, record_hash in video_pairs:
        video_distance[record_hash] = min(
            distance, video_distance.get(record_hash, distance)
        )

    selected = ""
    status = "unresolved"
    channel = ""
    if len(audio_candidates) == 1:
        selected = audio_candidates[0]
        status = "resolved_audio32_exact"
        channel = "audio32"
        if selected in video_distance:
            channel = "audio32+video24"
    elif len(audio_candidates) > 1:
        shared = sorted(set(audio_candidates) & set(video_distance))
        if shared:
            minimum = min(video_distance[value] for value in shared)
            nearest = [value for value in shared if video_distance[value] == minimum]
            if len(nearest) == 1:
                selected = nearest[0]
                status = "resolved_joint_disambiguation"
                channel = "audio32+video24"
            else:
                status = "ambiguous_joint_candidates"
        else:
            status = "ambiguous_audio32_bucket"
    else:
        exact = sorted(
            record_hash
            for record_hash, distance in video_distance.items()
            if distance == 0
        )
        if len(exact) == 1:
            selected = exact[0]
            status = "resolved_video24_exact"
            channel = "video24_exact"
        elif len(exact) > 1:
            status = "ambiguous_video24_exact_bucket"
        elif video_distance:
            status = "video24_approximate_without_audio"
        else:
            status = "no_candidate"

    record = dict(by_hash.get(selected, {}))
    return {
        "claim_record_hash": selected,
        "claim_channel": channel,
        "claim_resolution_status": status,
        "claim_resolved_authenticated": bool(
            selected
            and record.get("_signature_ok")
            and record.get("_locator_auth_ok")
        ),
        "claim_signature_ok": bool(record.get("_signature_ok", False)),
        "claim_signature_reason": str(record.get("_signature_reason", "")),
        "claim_locator_auth_ok": bool(record.get("_locator_auth_ok", False)),
        "claim_locator_auth_reason": str(record.get("_locator_auth_reason", "")),
        "claim_locator_hex": str(record.get("authenticated_claim_locator_hex", "")),
        "claim_version": record.get("claim_version", ""),
        "claim_family_sha256": str(record.get("claim_family_sha256", "")),
        "claim_source_id": str(record.get("source_id", "")),
        "claim_enroll_id": str(record.get("enroll_id", "")),
        "audio_candidate_count": len(audio_candidates),
        "video_candidate_count_radius2": len(video_distance),
        "video_exact_candidate_count": sum(
            distance == 0 for distance in video_distance.values()
        ),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    return {
        "rows": n,
        "authenticated_claim_resolved": sum(
            int(bool(row.get("claim_resolved_authenticated"))) for row in rows
        ),
        "expected_record_hit": sum(
            int(bool(row.get("claim_expected_hit"))) for row in rows
        ),
        "claim_final_ok": sum(int(bool(row.get("claim_final_ok"))) for row in rows),
        "wrong_source_selected": sum(
            int(bool(row.get("claim_wrong_source"))) for row in rows
        ),
        "unresolved": sum(
            int(not bool(row.get("claim_resolved_authenticated"))) for row in rows
        ),
        "audio_tag_recovered": sum(
            int(bool(row.get("audio_tag_recovered"))) for row in rows
        ),
        "video_pointer_recovered": sum(
            int(bool(row.get("video_pointer_recovered"))) for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--signed-payload-registry", type=Path, required=True)
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=Path("/path/to/runtime/anaconda3/envs/iap/bin/ffmpeg"),
    )
    parser.add_argument("--video-resize", type=int, default=256)
    parser.add_argument("--video-max-frames", type=int, default=32)
    parser.add_argument("--audio-master-key", default="tdsc-m7-audio-packet-beacon-v1")
    parser.add_argument("--audio-decode-window-sec", type=float, default=0.75)
    parser.add_argument("--audio-decode-hop-sec", type=float, default=0.1875)
    parser.add_argument("--audio-edge-guard-sec", type=float, default=0.2)
    args = parser.parse_args()

    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    by_hash, audio_index, video_index = load_registry(
        args.signed_payload_registry, args.public_key
    )
    masks = neighbor_masks(24, 2)

    from audioseal import AudioSeal
    import videoseal

    detector = AudioSeal.load_detector("audioseal_detector_16bits").to(device).eval()
    video_model = videoseal.load("videoseal")
    video_model.eval().to(device)

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(read_csv(args.tasks), start=1):
        media = Path(
            str(row.get("output_video_path") or row.get("embedded_video_path") or "")
        )
        base = {
            "task_id": row.get("task_id", ""),
            "sample_id": row.get("sample_id", ""),
            "enroll_id": row.get("enroll_id", ""),
            "source_id": row.get("source_id", ""),
            "generator": row.get("generator", ""),
            "cond": row.get("cond", ""),
            "m6_attack_type": row.get("m6_attack_type", ""),
            "m6_candidate_policy": row.get("m6_candidate_policy", ""),
            "m7_payload_registry_record_sha256": row.get(
                "m7_payload_registry_record_sha256", ""
            ),
            "media_path": str(media),
        }
        try:
            if not media.exists():
                raise FileNotFoundError(str(media))
            audio = recover_audio_claim(
                detector,
                media,
                args.out_dir / "decoded_audio" / f"{index:06d}.wav",
                args.ffmpeg,
                args.audio_master_key,
                args.audio_decode_window_sec,
                args.audio_decode_hop_sec,
                args.audio_edge_guard_sec,
                device,
            )
            video = recover_video_claim(
                video_model,
                media,
                24,
                args.video_resize,
                args.video_max_frames,
                device,
            )
            claim = resolve_claim(
                audio,
                video,
                by_hash,
                audio_index,
                video_index,
                masks,
            )
            expected = str(base["m7_payload_registry_record_sha256"])
            selected = str(claim.get("claim_record_hash", ""))
            expected_hit = bool(selected and expected and selected == expected)
            wrong_source = bool(
                selected
                and expected
                and str(claim.get("claim_source_id", "")) != str(base["source_id"])
            )
            source_scope_ok = bool(
                selected
                and str(claim.get("claim_source_id", "")) == str(base["source_id"])
            )
            enroll_scope_ok = bool(
                selected
                and str(claim.get("claim_enroll_id", "")) == str(base["enroll_id"])
            )
            final_ok = bool(
                claim.get("claim_resolved_authenticated")
                and expected_hit
                and source_scope_ok
                and enroll_scope_ok
            )
            rows.append(
                {
                    **base,
                    **audio,
                    **video,
                    **claim,
                    "claim_expected_hit": expected_hit,
                    "claim_wrong_source": wrong_source,
                    "claim_source_scope_ok": source_scope_ok,
                    "claim_enroll_scope_ok": enroll_scope_ok,
                    "claim_final_ok": final_ok,
                }
            )
            print(
                f"[m7-locator-v2-verify {index}] resolved={int(bool(claim.get('claim_resolved_authenticated')))} expected={int(expected_hit)} {base['sample_id']}",
                flush=True,
            )
        except Exception as exc:
            failures.append({**base, "error": f"{type(exc).__name__}: {exc}"})
            print(
                f"[m7-locator-v2-verify {index}] FAILED {base['sample_id']}: {exc}",
                flush=True,
            )

    rows_csv = args.out_dir / "m7_authenticated_locator_v2_claim_rows.csv"
    failures_csv = args.out_dir / "m7_authenticated_locator_v2_claim_failures.csv"
    summary_json = args.out_dir / "m7_authenticated_locator_v2_claim_summary.json"
    write_csv(rows_csv, rows)
    write_csv(failures_csv, failures)
    summary = {
        "policy": LOCATOR_POLICY,
        "tasks": str(args.tasks),
        "signed_payload_registry": str(args.signed_payload_registry),
        "public_key": str(args.public_key),
        **summarize(rows),
        "failures": len(failures),
        "outputs": {
            "rows_csv": str(rows_csv),
            "failures_csv": str(failures_csv),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
