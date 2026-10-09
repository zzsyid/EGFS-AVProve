#!/usr/bin/env bash
set -euo pipefail

TDSC_ROOT="${TDSC_ROOT:-/path/to/EGFS-AVProve}"
OUT="${OUT:-${TDSC_ROOT}/rq1_eval/artifacts_m4_windowed_registry_fusion}"
M3_FUSED="${M3_FUSED:-${TDSC_ROOT}/rq1_eval/artifacts_m3_checkpoint_freeze/m3_scale50_audio_window_v3_20260604/artifacts/m3_fused_rows.csv}"
AUDIO_SEGMENT_REGISTRY="${AUDIO_SEGMENT_REGISTRY:-${TDSC_ROOT}/rq1_eval/artifacts_segmented_evt_bch_splice200_v2/segmented_evt_bch_registry.jsonl}"
VIDEO_WINDOW_FEATURE_MANIFESTS="${VIDEO_WINDOW_FEATURE_MANIFESTS:-${TDSC_ROOT}/manifests/rq1_m2_video_splice_scale50_window_features.csv ${TDSC_ROOT}/manifests/rq1_m2_full_video_replacement_scale50_window_features.csv ${TDSC_ROOT}/manifests/rq1_m2_lavdf_modal_scale50_window_features.csv ${TDSC_ROOT}/manifests/rq1_m2_video_window_benign_holdout_features.csv}"
VIDEO_WINDOW_THRESHOLD_SUMMARIES="${VIDEO_WINDOW_THRESHOLD_SUMMARIES:-${TDSC_ROOT}/rq1_eval/artifacts_m2_video_splice_scale50/window_evaluation/m2_video_window_summary.json ${TDSC_ROOT}/rq1_eval/artifacts_m2_full_video_replacement_scale50/window_evaluation/m2_video_window_summary.json ${TDSC_ROOT}/rq1_eval/artifacts_m2_lavdf_modal_scale50/window_evaluation/m2_video_window_summary.json ${TDSC_ROOT}/rq1_eval/artifacts_m2_video_window_benign_holdout/evaluation/m2_video_window_summary.json}"

cd "${TDSC_ROOT}"
mkdir -p "${OUT}"

# shellcheck disable=SC2206
video_window_feature_manifests=(${VIDEO_WINDOW_FEATURE_MANIFESTS})
# shellcheck disable=SC2206
video_window_threshold_summaries=(${VIDEO_WINDOW_THRESHOLD_SUMMARIES})

python rq1_eval/build_m4_video_window_commitment_registry.py \
  --window-feature-manifests "${video_window_feature_manifests[@]}" \
  --threshold-summaries "${video_window_threshold_summaries[@]}" \
  --out-registry "${OUT}/video_window_commitment_registry.jsonl" \
  --out-summary "${OUT}/video_window_commitment_registry_summary.json"

python rq1_eval/build_m4_windowed_system_registry.py \
  --m3-fused "${M3_FUSED}" \
  --audio-segment-registry "${AUDIO_SEGMENT_REGISTRY}" \
  --video-window-registry "${OUT}/video_window_commitment_registry.jsonl" \
  --out-registry "${OUT}/windowed_system_registry.jsonl" \
  --out-summary "${OUT}/windowed_system_registry_summary.json"

if [[ ! -f "${OUT}/private.pem" || ! -f "${OUT}/public.pem" ]]; then
  python rq1_eval/registry_signatures.py keygen \
    --private-key "${OUT}/private.pem" \
    --public-key "${OUT}/public.pem"
fi

python rq1_eval/registry_signatures.py sign \
  --registry "${OUT}/windowed_system_registry.jsonl" \
  --private-key "${OUT}/private.pem" \
  --out "${OUT}/signed_windowed_system_registry.jsonl" \
  --policy-version "tdsc-windowed-system-registry-v1"

python rq1_eval/registry_signatures.py verify \
  --signed-registry "${OUT}/signed_windowed_system_registry.jsonl" \
  --public-key "${OUT}/public.pem"

python rq1_eval/verify_m4_windowed_registry_fusion.py \
  --signed-registry "${OUT}/signed_windowed_system_registry.jsonl" \
  --public-key "${OUT}/public.pem" \
  --m3-fused "${M3_FUSED}" \
  --out-dir "${OUT}/verification" \
  --state-col state_full_av

echo "[m4-windowed-registry-fusion] wrote ${OUT}"
