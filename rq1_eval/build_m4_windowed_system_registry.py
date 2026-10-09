#!/usr/bin/env python3
"""Build a signed-registry-ready manifest for the window-aware M4 verifier.

This registry binds a source claim to the semantic-commitment evidence used by
frozen M3. It is intentionally system-level: the audio segmented BCH registry
already contains helper data and digests; video-window evidence is represented
as a policy/checkpoint component until video-window helper records are
materialized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from registry_signatures import canonical


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + '\n')


@lru_cache(maxsize=None)
def sha256_file(path: Path) -> str:
    if not path.exists():
        return ''
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def merkle_root(leaves: List[bytes]) -> str:
    if not leaves:
        return hashlib.sha256(b'tdsc-empty-windowed-system-registry').hexdigest()
    nodes = [hashlib.sha256(b'\x00' + leaf).digest() for leaf in leaves]
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [
            hashlib.sha256(b'\x01' + nodes[i] + nodes[i + 1]).digest()
            for i in range(0, len(nodes), 2)
        ]
    return nodes[0].hex()


def first_nonempty(values: Iterable[object]) -> str:
    for value in values:
        text = str(value)
        if text and text.lower() not in {'nan', 'none'}:
            return text
    return ''


def audio_registry_map(paths: List[Path]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for path in paths:
        for record in read_jsonl(path):
            enroll_id = str(record.get('enroll_id', ''))
            if not enroll_id or enroll_id in out:
                continue
            out[enroll_id] = {
                'component': 'audio_segmented_bch',
                'registry_path': str(path),
                'registry_sha256': sha256_file(path),
                'merkle_root': str(record.get('merkle_root', '')),
                'segment_count': len(record.get('segments', [])),
                'n': int(record.get('segments', [{}])[0].get('n', 0)) if record.get('segments') else 0,
                't': int(record.get('segments', [{}])[0].get('t', 0)) if record.get('segments') else 0,
                'helper_data': 'public p_hex per window; digest D_hex; nonce_hex',
            }
    return out


def video_window_registry_map(paths: List[Path]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for path in paths:
        for record in read_jsonl(path):
            enroll_id = str(record.get('enroll_id', ''))
            if not enroll_id or enroll_id in out:
                continue
            segments = record.get('segments', [])
            secure_segments = sum(1 for segment in segments if bool(segment.get('secure')))
            out[enroll_id] = {
                'component': 'video_window_commitment',
                'registry_path': str(path),
                'registry_sha256': sha256_file(path),
                'merkle_root': str(record.get('merkle_root', '')),
                'window_count': int(record.get('window_count', 0)),
                'segment_count': len(segments),
                'secure_segments': int(secure_segments),
                'fallback_segments': int(len(segments) - secure_segments),
                'thresholds_bits': record.get('policy', {}).get('thresholds_bits', {}),
                'helper_data': 'public p_hex per video window/channel; digest D_hex; nonce_hex',
                'helper_data_status': 'materialized',
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m3-fused', type=Path, required=True)
    parser.add_argument('--out-registry', type=Path, required=True)
    parser.add_argument('--out-summary', type=Path, default=None)
    parser.add_argument('--audio-segment-registry', type=Path, action='append', default=[])
    parser.add_argument('--video-window-registry', type=Path, action='append', default=[])
    parser.add_argument('--m3-checkpoint', default='m3_scale50_audio_window_v3_20260604')
    parser.add_argument('--policy-version', default='tdsc-windowed-system-registry-v1')
    parser.add_argument('--video-window-policy', default='m2-video-window-stage0-policy-reference')
    args = parser.parse_args()

    m3 = pd.read_csv(args.m3_fused).fillna('')
    required = {'enroll_id', 'source_id', 'sample_id'}
    missing = required - set(m3.columns)
    if missing:
        raise ValueError(f'M3 fused rows missing columns: {sorted(missing)}')

    audio_by_enroll = audio_registry_map(args.audio_segment_registry)
    video_window_by_enroll = video_window_registry_map(args.video_window_registry)
    records: List[Dict] = []
    for enroll_id, group in m3.groupby('enroll_id', sort=False):
        source_id = first_nonempty(group['source_id'].tolist())
        sample_id = first_nonempty(group[group['population'].astype(str).str.lower().eq('orig')]['sample_id'].tolist())
        if not sample_id:
            sample_id = first_nonempty(group['sample_id'].tolist())
        components = [
            {
                'component': 'm3_fused_policy',
                'checkpoint': args.m3_checkpoint,
                'm3_fused_path': str(args.m3_fused),
                'm3_fused_sha256': sha256_file(args.m3_fused),
                'state_column': 'state_full_av',
                'scope': 'signed verifier emits this frozen fusion state after source-scope and signature checks',
            },
            {
                'component': 'audio_whole_vector_commitment',
                'role': 'identity / coarse event evidence',
                'm3_state_column': 'state_audio_id',
            },
            {
                'component': 'video_whole_commitment',
                'role': 'coarse visual evidence',
                'm3_state_column': 'state_video_whole',
            },
            video_window_by_enroll.get(str(enroll_id), {
                'component': 'video_window_commitment',
                'role': 'local visual tamper evidence',
                'policy_reference': args.video_window_policy,
                'm3_state_column': 'state_video_window',
                'helper_data_status': 'missing_for_this_enrollment',
            }),
        ]
        if str(enroll_id) in audio_by_enroll:
            components.append(audio_by_enroll[str(enroll_id)])
        else:
            components.append({
                'component': 'audio_segmented_bch',
                'status': 'missing_for_this_enrollment',
                'm3_state_column': 'state_audio_window',
            })
        record = {
            'registry_schema': 'tdsc-windowed-system-registry-v1',
            'record_type': 'windowed_system_semantic_registry',
            'policy_version': args.policy_version,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'source_id': str(source_id),
            'sample_id': str(sample_id),
            'enroll_id': str(enroll_id),
            'm3_checkpoint': args.m3_checkpoint,
            'components': components,
            'component_count': len(components),
        }
        record['merkle_root'] = merkle_root([canonical(component) for component in components])
        record['registry_record_sha256'] = hashlib.sha256(canonical(record)).hexdigest()
        records.append(record)

    write_jsonl(args.out_registry, records)
    with_audio = sum(
        any(c.get('component') == 'audio_segmented_bch' and c.get('merkle_root') for c in r['components'])
        for r in records
    )
    with_video_window = sum(
        any(c.get('component') == 'video_window_commitment' and c.get('merkle_root') for c in r['components'])
        for r in records
    )
    video_secure_segments = sum(
        int(c.get('secure_segments', 0))
        for r in records
        for c in r['components']
        if c.get('component') == 'video_window_commitment'
    )
    video_fallback_segments = sum(
        int(c.get('fallback_segments', 0))
        for r in records
        for c in r['components']
        if c.get('component') == 'video_window_commitment'
    )
    summary = {
        'registry': str(args.out_registry),
        'm3_fused': str(args.m3_fused),
        'records': len(records),
        'audio_segmented_component_present': with_audio,
        'audio_segmented_component_missing': len(records) - with_audio,
        'video_window_component_present': with_video_window,
        'video_window_component_missing': len(records) - with_video_window,
        'video_window_secure_segments': video_secure_segments,
        'video_window_fallback_segments': video_fallback_segments,
        'policy_version': args.policy_version,
        'm3_checkpoint': args.m3_checkpoint,
        'video_window_status': 'materialized-when-feature-registry-present',
    }
    out_summary = args.out_summary or args.out_registry.with_suffix('.summary.json')
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
