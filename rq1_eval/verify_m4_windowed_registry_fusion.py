#!/usr/bin/env python3
"""Verify signed window-aware registry records and emit M3 fused provenance state."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from registry_signatures import canonical, load_public, read_jsonl, unsigned


STATE_VERIFIED = 'source-and-content verified'
STATE_MODAL = 'tampered/modal-replacement'
LEGACY_STATE_MODAL = 'modal-replacement'
STATE_SIGNATURE_FAILED = 'signature-failed'
STATE_REJECTED = 'rejected'
LEGACY_STATE_REJECTED = 'unknown'


def verify_signature(public: Any, record: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        payload_record = unsigned(record)
        payload = canonical(payload_record)
        if hashlib.sha256(payload).hexdigest() != record.get('record_sha256'):
            return False, 'record_sha256_mismatch'
        signature = base64.b64decode(record['signature_ed25519_b64'], validate=True)
        public.verify(signature, payload)
        return True, 'signature_ok'
    except Exception as exc:
        return False, f'signature_exception:{type(exc).__name__}'


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def is_l1(row: pd.Series) -> bool:
    return str(row.get('cond', '')).lower() == 'l1' or str(row.get('attack_level', '')).upper() == 'L1'


def is_attack(row: pd.Series) -> bool:
    population = str(row.get('population', '')).lower()
    cond = str(row.get('cond', '')).lower()
    level = str(row.get('attack_level', '')).upper()
    if population == 'orig' or cond == 'clean' or is_l1(row):
        return False
    return population in {'tamper', 'attack', 'different', 'swap'} or level in {'L3', 'L4'}


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    group_cols = ['split', 'population', 'cond', 'generator', 'attack_level', 'attack_family']
    out: List[Dict[str, object]] = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(group)
        out.append({
            **dict(zip(group_cols, keys)),
            'n': int(n),
            'source_and_content_verified': int((group['final_state'] == STATE_VERIFIED).sum()),
            'content_unverified': int((group['final_state'] == 'content-unverified').sum()),
            'localized_tamper': int((group['final_state'] == 'localized-tamper').sum()),
            'modal_replacement': int((group['final_state'] == STATE_MODAL).sum()),
            'rejected': int((group['final_state'] == STATE_REJECTED).sum()),
            'signature_failed': int((group['final_state'] == STATE_SIGNATURE_FAILED).sum()),
            'm3_parity_mismatch': int((~group['m3_parity']).sum()),
        })
    return out


def compact_security_summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    attack = df[df['is_attack_eval']]
    benign = df[df['is_l1_benign']]
    by_generator = {}
    for (split, generator, cond), group in attack.groupby(['split', 'generator', 'cond'], dropna=False):
        by_generator[f'{split}::{generator}::{cond}'] = {
            'n': int(len(group)),
            'false_verified': int((group['final_state'] == STATE_VERIFIED).sum()),
        }
    benign_by_generator = {}
    for (split, generator, cond), group in benign.groupby(['split', 'generator', 'cond'], dropna=False):
        benign_by_generator[f'{split}::{generator}::{cond}'] = {
            'n': int(len(group)),
            'verified': int((group['final_state'] == STATE_VERIFIED).sum()),
        }
    return {
        'rows': int(len(df)),
        'm3_parity_mismatches': int((~df['m3_parity']).sum()),
        'signature_failed': int((df['final_state'] == STATE_SIGNATURE_FAILED).sum()),
        'attacks': by_generator,
        'benign_l1': benign_by_generator,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--signed-registry', type=Path, required=True)
    parser.add_argument('--public-key', type=Path, required=True)
    parser.add_argument('--m3-fused', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--state-col', default='state_full_av')
    args = parser.parse_args()

    public = load_public(args.public_key)
    signed = read_jsonl(args.signed_registry)
    registry_by_enroll: Dict[str, Dict[str, Any]] = {}
    signature_failures = 0
    for envelope in signed:
        ok, reason = verify_signature(public, envelope)
        manifest = dict(envelope.get('manifest', {}))
        manifest['_signature_ok'] = ok
        manifest['_signature_reason'] = reason
        signature_failures += int(not ok)
        registry_by_enroll[str(manifest.get('enroll_id', ''))] = manifest

    m3 = pd.read_csv(args.m3_fused).fillna('')
    if args.state_col not in m3.columns:
        raise ValueError(f'M3 fused CSV missing state column {args.state_col}')

    rows: List[Dict[str, object]] = []
    skipped = {'missing_registry': 0}
    for _, row in m3.iterrows():
        enroll_id = str(row.get('enroll_id', ''))
        record = registry_by_enroll.get(enroll_id)
        if record is None:
            skipped['missing_registry'] += 1
            continue
        signature_ok = bool(record.get('_signature_ok', False))
        source_scope_ok = (
            str(record.get('source_id', '')) == str(row.get('source_id', ''))
            and str(record.get('enroll_id', '')) == enroll_id
        )
        m3_state = str(row.get(args.state_col, STATE_REJECTED) or STATE_REJECTED)
        if m3_state == LEGACY_STATE_MODAL:
            m3_state = STATE_MODAL
        elif m3_state == LEGACY_STATE_REJECTED:
            m3_state = STATE_REJECTED
        if not signature_ok:
            final_state = STATE_SIGNATURE_FAILED
            reason = str(record.get('_signature_reason', 'signature_failed'))
        elif not source_scope_ok:
            final_state = STATE_MODAL
            reason = 'signed_source_scope_mismatch'
        else:
            final_state = m3_state
            reason = 'signature_source_scope_and_windowed_m3_commitments_passed_to_fusion'
        components = record.get('components', [])
        has_video_window_materialized = any(
            c.get('component') == 'video_window_commitment' and c.get('merkle_root')
            for c in components
        )
        rows.append({
            'sample_id': str(row.get('sample_id', '')),
            'enroll_id': enroll_id,
            'source_id': str(row.get('source_id', '')),
            'split': str(row.get('split', '')),
            'population': str(row.get('population', '')),
            'cond': str(row.get('cond', '')),
            'generator': str(row.get('generator', '')),
            'attack_level': str(row.get('attack_level', '')),
            'attack_family': str(row.get('attack_family', '')),
            'signature_ok': signature_ok,
            'signature_reason': str(record.get('_signature_reason', '')),
            'source_scope_ok': source_scope_ok,
            'component_count': int(record.get('component_count', len(components))),
            'has_audio_segmented_bch': any(
                c.get('component') == 'audio_segmented_bch' and c.get('merkle_root')
                for c in components
            ),
            'has_video_window_policy': any(c.get('component') == 'video_window_commitment' for c in components),
            'has_video_window_materialized': has_video_window_materialized,
            'm3_state': m3_state,
            'final_state': final_state,
            'm3_parity': final_state == m3_state,
            'decision_reason': reason,
            'is_attack_eval': is_attack(row),
            'is_l1_benign': is_l1(row),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / 'm4_windowed_registry_fusion_rows.csv'
    summary_path = args.out_dir / 'm4_windowed_registry_fusion_summary.csv'
    json_path = args.out_dir / 'm4_windowed_registry_fusion_summary.json'
    write_csv(rows_path, rows)
    write_csv(summary_path, summarize(rows))
    report = {
        'signed_registry': str(args.signed_registry),
        'public_key': str(args.public_key),
        'm3_fused': str(args.m3_fused),
        'state_col': args.state_col,
        'registry_records': len(signed),
        'm3_rows': int(len(m3)),
        'checked_rows': len(rows),
        'signature_failures': signature_failures,
        'video_window_materialized_rows': sum(bool(row.get('has_video_window_materialized')) for row in rows),
        'video_window_missing_materialized_rows': sum(not bool(row.get('has_video_window_materialized')) for row in rows),
        'skipped': skipped,
        'compact_security_summary': compact_security_summary(rows),
        'outputs': {
            'rows_csv': str(rows_path),
            'summary_csv': str(summary_path),
            'summary_json': str(json_path),
        },
        'scope_note': (
            'This verifier is the signed-registry wrapper around frozen M3 windowed evidence. '
            'Audio segmented BCH helper roots and materialized video-window helper roots are bound '
            'when available; remaining work is full video-window coverage and replacing RadiusECC '
            'fallback segments with a stronger feasible code/quantizer.'
        ),
    }
    json_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
