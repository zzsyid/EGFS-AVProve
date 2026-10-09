#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/path/to/EGFS-AVProve}"
OUT="${OUT:?Set OUT to an artifacts_m7_m6_e2e_* directory}"
IAP_PYTHON="${IAP_PYTHON:-/path/to/runtime/anaconda3/envs/iap/bin/python}"
METASEAL_PYTHON="${METASEAL_PYTHON:-/path/to/runtime/anaconda3/envs/metaseal/bin/python}"
FFMPEG="${FFMPEG:-/path/to/runtime/anaconda3/envs/iap/bin/ffmpeg}"
DEVICES="${DEVICES:-cuda:0}"
DEVICE="${DEVICE:-cuda:0}"
SIGNED_M4_REGISTRY="${SIGNED_M4_REGISTRY:-${ROOT}/rq1_eval/artifacts_m4_windowed_registry_fusion/signed_windowed_system_registry.jsonl}"
SIGNED_M4_PUBLIC_KEY="${SIGNED_M4_PUBLIC_KEY:-${ROOT}/rq1_eval/artifacts_m4_windowed_registry_fusion/public.pem}"

cd "${ROOT}"
export PATH=/path/to/runtime/anaconda3/envs/iap/bin:${PATH}

TASKS="${OUT}/m7_m6_rewritten_tasks.csv"
COMPLETED="${OUT}/m7_m6_completed_inventory.csv"
SIGNED_PAYLOAD_REGISTRY="${OUT}/signed_m7_authenticated_locator_registry.jsonl"
PAYLOAD_PUBLIC_KEY="${OUT}/public.pem"

if [[ ! -f "${TASKS}" ]]; then
  echo "[m7-m6-verify] missing tasks: ${TASKS}" >&2
  exit 2
fi
if [[ ! -f "${COMPLETED}" ]]; then
  echo "[m7-m6-verify] missing completed inventory: ${COMPLETED}" >&2
  exit 2
fi

echo "[m7-m6-verify] authenticated claim recovery -> ${OUT}/claim_recovery"
"${METASEAL_PYTHON}" rq1_eval/verify_m7_authenticated_claim_locator_v2.py \
  --tasks "${TASKS}" \
  --signed-payload-registry "${SIGNED_PAYLOAD_REGISTRY}" \
  --public-key "${PAYLOAD_PUBLIC_KEY}" \
  --out-dir "${OUT}/claim_recovery" \
  --device "${DEVICE}" \
  --ffmpeg "${FFMPEG}"

echo "[m7-m6-verify] extract id-audio"
"${IAP_PYTHON}" rq1_eval/extract_id_audio_features.py \
  --manifest "${COMPLETED}" \
  --out-root "${OUT}/features/id_audio" \
  --out-manifest "${OUT}/features/id_audio_features.csv" \
  --devices "${DEVICES}" \
  --resume

echo "[m7-m6-verify] extract video whole"
"${IAP_PYTHON}" rq1_eval/extract_video_stage0_features.py \
  --manifest "${COMPLETED}" \
  --out-root "${OUT}/features/video" \
  --out-manifest "${OUT}/features/video_features.csv" \
  --devices "${DEVICES}" \
  --resume

echo "[m7-m6-verify] extract video window"
"${IAP_PYTHON}" rq1_eval/extract_video_window_stage0_features.py \
  --manifest "${COMPLETED}" \
  --out-root "${OUT}/features/video_window" \
  --out-manifest "${OUT}/features/video_window_features.csv" \
  --devices "${DEVICES}" \
  --resume \
  --keep-all

echo "[m7-m6-verify] extract audio window"
"${IAP_PYTHON}" rq1_eval/extract_evt_audio_v2.py \
  --manifest "${COMPLETED}" \
  --out-root "${OUT}/features/evt_audio_span05_hop025" \
  --out-manifest "${OUT}/features/evt_audio_span05_hop025_features.csv" \
  --devices "${DEVICES}" \
  --resume \
  --whisper-model small \
  --window-span-sec 0.5 \
  --window-hop-sec 0.25 \
  --window-batch-size 16

echo "[m7-m6-verify] extract clean source id-audio"
"${IAP_PYTHON}" rq1_eval/extract_id_audio_features.py \
  --manifest "${OUT}/m7_m6_clean_manifest.csv" \
  --out-root "${OUT}/features/source_clean/id_audio" \
  --out-manifest "${OUT}/features/source_clean/id_audio_features.csv" \
  --devices "${DEVICES}" \
  --resume

echo "[m7-m6-verify] extract clean source video whole"
"${IAP_PYTHON}" rq1_eval/extract_video_stage0_features.py \
  --manifest "${OUT}/m7_m6_clean_manifest.csv" \
  --out-root "${OUT}/features/source_clean/video" \
  --out-manifest "${OUT}/features/source_clean/video_features.csv" \
  --devices "${DEVICES}" \
  --resume

echo "[m7-m6-verify] extract clean source video window"
"${IAP_PYTHON}" rq1_eval/extract_video_window_stage0_features.py \
  --manifest "${OUT}/m7_m6_clean_manifest.csv" \
  --out-root "${OUT}/features/source_clean/video_window" \
  --out-manifest "${OUT}/features/source_clean/video_window_features.csv" \
  --devices "${DEVICES}" \
  --resume \
  --keep-all

echo "[m7-m6-verify] extract clean source audio window"
"${IAP_PYTHON}" rq1_eval/extract_evt_audio_v2.py \
  --manifest "${OUT}/m7_m6_clean_manifest.csv" \
  --out-root "${OUT}/features/source_clean/evt_audio_span05_hop025" \
  --out-manifest "${OUT}/features/source_clean/evt_audio_span05_hop025_features.csv" \
  --devices "${DEVICES}" \
  --resume \
  --whisper-model small \
  --window-span-sec 0.5 \
  --window-hop-sec 0.25 \
  --window-batch-size 16

echo "[m7-m6-verify] build combined feature manifests"
"${IAP_PYTHON}" rq1_eval/build_m6_verification_manifests.py \
  --completed-inventory "${COMPLETED}" \
  --m6-id-audio "${OUT}/features/id_audio_features.csv" \
  --m6-video "${OUT}/features/video_features.csv" \
  --m6-video-window "${OUT}/features/video_window_features.csv" \
  --m6-evt-audio "${OUT}/features/evt_audio_span05_hop025_features.csv" \
  --id-audio-baseline "${OUT}/features/source_clean/id_audio_features.csv" \
  --video-baseline "${OUT}/features/source_clean/video_features.csv" \
  --video-window-baseline "${OUT}/features/source_clean/video_window_features.csv" \
  --evt-audio-baseline "${OUT}/features/source_clean/evt_audio_span05_hop025_features.csv" \
  --out-id-audio "${OUT}/features/id_audio_features_combined.csv" \
  --out-video "${OUT}/features/video_features_combined.csv" \
  --out-video-window "${OUT}/features/video_window_features_combined.csv" \
  --out-evt-audio "${OUT}/features/evt_audio_span05_hop025_features_combined.csv"

echo "[m7-m6-verify] evaluate M1/M2/audio-window"
"${IAP_PYTHON}" rq1_eval/evaluate_m1_id_audio_stage0.py \
  --feature-manifest "${OUT}/features/id_audio_features_combined.csv" \
  --pilot-manifest "${COMPLETED}" \
  --feature-failures "${OUT}/features/id_audio_features.failures.csv" \
  --out-dir "${OUT}/id_audio_evaluation" \
  --drop-missing-orig
"${IAP_PYTHON}" rq1_eval/evaluate_m2_video_stage0.py \
  --feature-manifest "${OUT}/features/video_features_combined.csv" \
  --pilot-manifest "${COMPLETED}" \
  --feature-failures "${OUT}/features/video_features.failures.csv" \
  --out-dir "${OUT}/video_evaluation" \
  --drop-missing-orig
"${IAP_PYTHON}" rq1_eval/evaluate_m2_video_window_stage0.py \
  --feature-manifest "${OUT}/features/video_window_features_combined.csv" \
  --out-dir "${OUT}/video_window_evaluation"
"${IAP_PYTHON}" rq1_eval/evaluate_segmented_evt_bch.py \
  --manifest "${OUT}/features/evt_audio_span05_hop025_features_combined.csv" \
  --out-dir "${OUT}/audio_window_span05_hop025" \
  --bch-t 54 \
  --policies 1.0 0.95 0.90 \
  --tamper-run-threshold 1 \
  --drop-orphans

echo "[m7-m6-verify] fuse M3"
"${IAP_PYTHON}" rq1_eval/fuse_m3_av_states.py \
  --audio-event-long "${OUT}/audio_window_span05_hop025/segmented_evt_bch_scores.csv" \
  --audio-event-short "${OUT}/audio_window_span05_hop025/segmented_evt_bch_scores.csv" \
  --audio-id "${OUT}/id_audio_evaluation/m1_id_audio_decisions.csv" \
  --audio-window "${OUT}/audio_window_span05_hop025/segmented_evt_bch_scores.csv" \
  --video-whole "${OUT}/video_evaluation/m2_video_decisions.csv" \
  --video-window "${OUT}/video_window_evaluation/m2_video_window_decisions.csv" \
  --out-dir "${OUT}/m3_fusion" \
  --policy v3 \
  --audio-window-long-run 7 \
  --audio-window-short-run 2 \
  --audio-window-max-failed 4 \
  --audio-window-ham-threshold 80 \
  --audio-window-q90-threshold 70

"${IAP_PYTHON}" rq1_eval/enrich_m6_metadata_from_inventory.py \
  --input "${OUT}/m3_fusion/m3_fused_rows.csv" \
  --inventory "${TASKS}" \
  --out "${OUT}/m3_fusion/m3_fused_rows_m6_m7_enriched.csv"

echo "[m7-m6-verify] signed M4 verification"
"${IAP_PYTHON}" rq1_eval/verify_m6_signed_m4_media_realized.py \
  --signed-registry "${SIGNED_M4_REGISTRY}" \
  --public-key "${SIGNED_M4_PUBLIC_KEY}" \
  --m3-fused "${OUT}/m3_fusion/m3_fused_rows_m6_m7_enriched.csv" \
  --out-dir "${OUT}/signed_m4_verification" \
  --m6-only \
  --require-m6-relevant-modality

echo "[m7-m6-verify] complete -> ${OUT}"
