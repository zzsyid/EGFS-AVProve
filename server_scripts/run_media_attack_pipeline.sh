#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/path/to/EGFS-AVProve}"
IAP_PYTHON="${IAP_PYTHON:-/path/to/runtime/anaconda3/envs/iap/bin/python}"
METASEAL_PYTHON="${METASEAL_PYTHON:-/path/to/runtime/anaconda3/envs/metaseal/bin/python}"
FFMPEG="${FFMPEG:-/path/to/runtime/anaconda3/envs/iap/bin/ffmpeg}"
FFPROBE="${FFPROBE:-/path/to/runtime/anaconda3/envs/iap/bin/ffprobe}"

M6_TASKS="${M6_TASKS:-${ROOT}/manifests/rq1_m6_media_adaptive_tasks.csv}"
M4_DIR="${M4_DIR:-${ROOT}/rq1_eval/artifacts_m4_windowed_registry_fusion}"
SIGNED_M4_REGISTRY="${SIGNED_M4_REGISTRY:-${M4_DIR}/signed_windowed_system_registry.jsonl}"

OUT="${OUT:-${ROOT}/rq1_eval/artifacts_media_attack_pipeline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/datasets/egfs_media_attacks}"
GENERATORS="${GENERATORS:-ffmpeg_splice,ffmpeg_video_splice,ffmpeg_full_video_replacement}"
LIMIT="${LIMIT:-0}"
PER_GENERATOR_LIMIT="${PER_GENERATOR_LIMIT:-0}"
SPLITS="${SPLITS:-}"
DEVICE="${DEVICE:-cuda:0}"
VIDEO_BATCH_FRAMES="${VIDEO_BATCH_FRAMES:-8}"
VIDEO_RESIZE="${VIDEO_RESIZE:-256}"
RUN_ATTACKS="${RUN_ATTACKS:-1}"
WRAPPER_CONFIG="${WRAPPER_CONFIG:-}"
WRAPPER_GENERATORS="${WRAPPER_GENERATORS:-ffmpeg_splice ffmpeg_video_splice ffmpeg_full_video_replacement}"
WORK_ROOT="${WORK_ROOT:-rq1_wrapper_work_m7_m6}"

cd "${ROOT}"
mkdir -p "${OUT}" manifests

SELECTED_CLEAN="${OUT}/m7_m6_clean_manifest.csv"
SELECTED_TASKS="${OUT}/m7_m6_selected_tasks.csv"
SELECTED_SUMMARY="${OUT}/m7_m6_prepare_clean_summary.json"
PAYLOAD_MANIFEST="${OUT}/m7_payload_manifest.csv"
BASE_PAYLOAD_REGISTRY="${OUT}/m7_base_payload_registry.jsonl"
SIGNED_BASE_PAYLOAD_REGISTRY="${OUT}/signed_m7_base_payload_registry.jsonl"
AUTH_PAYLOAD_REGISTRY="${OUT}/m7_authenticated_locator_registry.jsonl"
AUTH_MAPPING="${OUT}/m7_authenticated_locator_mapping.csv"
SIGNED_PAYLOAD_REGISTRY="${OUT}/signed_m7_authenticated_locator_registry.jsonl"
PRIVATE_KEY="${OUT}/private.pem"
PUBLIC_KEY="${OUT}/public.pem"
EMBEDDED_MANIFEST="${OUT}/m7_authenticated_locator_embedded_manifest.csv"
REWRITTEN_TASKS="${OUT}/m7_m6_rewritten_tasks.csv"
REWRITTEN_MISSING="${OUT}/m7_m6_rewritten_missing.csv"
COMPLETED="${OUT}/m7_m6_completed_inventory.csv"

echo "[m7-m6-e2e] prepare selected M6 sources"
"${IAP_PYTHON}" rq1_eval/prepare_m7_m6_e2e_inputs.py \
  --m6-tasks "${M6_TASKS}" \
  --generators "${GENERATORS}" \
  --limit "${LIMIT}" \
  --per-generator-limit "${PER_GENERATOR_LIMIT}" \
  --splits "${SPLITS}" \
  --out-clean-manifest "${SELECTED_CLEAN}" \
  --out-selected-tasks "${SELECTED_TASKS}" \
  --out-summary "${SELECTED_SUMMARY}"

echo "[m7-m6-e2e] build M7 payload registry for selected M6 sources"
"${IAP_PYTHON}" rq1_eval/build_m7_payload_registry.py \
  --clean-manifest "${SELECTED_CLEAN}" \
  --signed-m4-registry "${SIGNED_M4_REGISTRY}" \
  --out-manifest "${PAYLOAD_MANIFEST}" \
  --out-registry "${BASE_PAYLOAD_REGISTRY}" \
  --out-summary "${OUT}/m7_payload_registry_summary.json" \
  --allow-missing-m4

if [[ ! -f "${PRIVATE_KEY}" || ! -f "${PUBLIC_KEY}" ]]; then
  "${METASEAL_PYTHON}" rq1_eval/registry_signatures.py keygen \
    --private-key "${PRIVATE_KEY}" \
    --public-key "${PUBLIC_KEY}"
fi

echo "[m7-m6-e2e] sign base payload registry"
"${METASEAL_PYTHON}" rq1_eval/registry_signatures.py sign \
  --registry "${BASE_PAYLOAD_REGISTRY}" \
  --private-key "${PRIVATE_KEY}" \
  --out "${SIGNED_BASE_PAYLOAD_REGISTRY}" \
  --policy-version "tdsc-m7-payload-base-v1"
"${METASEAL_PYTHON}" rq1_eval/registry_signatures.py verify \
  --signed-registry "${SIGNED_BASE_PAYLOAD_REGISTRY}" \
  --public-key "${PUBLIC_KEY}"

echo "[m7-m6-e2e] build and sign authenticated locator registry"
"${METASEAL_PYTHON}" rq1_eval/build_m7_authenticated_claim_registry.py \
  --part "${SIGNED_BASE_PAYLOAD_REGISTRY}" "${PUBLIC_KEY}" \
  --out-registry "${AUTH_PAYLOAD_REGISTRY}" \
  --out-mapping "${AUTH_MAPPING}" \
  --out-summary "${OUT}/m7_authenticated_locator_registry_summary.json"
"${METASEAL_PYTHON}" rq1_eval/registry_signatures.py sign \
  --registry "${AUTH_PAYLOAD_REGISTRY}" \
  --private-key "${PRIVATE_KEY}" \
  --out "${SIGNED_PAYLOAD_REGISTRY}" \
  --policy-version "tdsc-m7-authenticated-claim-locator-v2"
"${METASEAL_PYTHON}" rq1_eval/registry_signatures.py verify \
  --signed-registry "${SIGNED_PAYLOAD_REGISTRY}" \
  --public-key "${PUBLIC_KEY}"

echo "[m7-m6-e2e] materialize authenticated Audio32/Video24 locators"
"${METASEAL_PYTHON}" rq1_eval/materialize_m7_authenticated_claim_locator_v2.py \
  --signed-payload-registry "${SIGNED_PAYLOAD_REGISTRY}" \
  --public-key "${PUBLIC_KEY}" \
  --out-root "${OUT}/embedded" \
  --out-manifest "${EMBEDDED_MANIFEST}" \
  --device "${DEVICE}" \
  --ffmpeg "${FFMPEG}" \
  --video-resize "${VIDEO_RESIZE}" \
  --video-batch-frames "${VIDEO_BATCH_FRAMES}" \
  --resume

echo "[m7-m6-e2e] bind selected attacks to authenticated locator media"
"${IAP_PYTHON}" rq1_eval/prepare_m7_authenticated_locator_v2_attack_tasks.py \
  --tasks "${SELECTED_TASKS}" \
  --mapping "${AUTH_MAPPING}" \
  --embedded-manifest "${EMBEDDED_MANIFEST}" \
  --signed-registry "${SIGNED_PAYLOAD_REGISTRY}" \
  --public-key "${PUBLIC_KEY}" \
  --out-tasks "${REWRITTEN_TASKS}" \
  --out-missing "${REWRITTEN_MISSING}" \
  --output-root "${OUTPUT_ROOT}" \
  --out-summary "${OUT}/m7_m6_rewrite_summary.json"

if [[ "${RUN_ATTACKS}" == "1" ]]; then
  echo "[m7-m6-e2e] run M6 attacks on M7 embedded media: ${WRAPPER_GENERATORS}"
  WRAPPER_ARGS=(--tasks "${REWRITTEN_TASKS}")
  if [[ -n "${WRAPPER_CONFIG}" ]]; then
    WRAPPER_ARGS+=(--config "${WRAPPER_CONFIG}")
  fi
  "${IAP_PYTHON}" rq1_eval/run_attack_wrappers.py \
    "${WRAPPER_ARGS[@]}" \
    --generators ${WRAPPER_GENERATORS} \
    --resume \
    --device "${DEVICE}" \
    --ffmpeg "${FFMPEG}" \
    --ffprobe "${FFPROBE}" \
    --work-root "${WORK_ROOT}" \
    --failures-csv "${OUT}/m7_m6_wrapper.failures.csv"
else
  echo "[m7-m6-e2e] skip wrapper execution because RUN_ATTACKS=${RUN_ATTACKS}"
fi

echo "[m7-m6-e2e] collect completed attack inventory"
"${IAP_PYTHON}" rq1_eval/collect_completed_attack_inventory.py \
  --tasks "${REWRITTEN_TASKS}" \
  --out "${COMPLETED}" \
  --allow-missing

cat > "${OUT}/run_summary.json" <<JSON
{
  "scope": "authenticated Audio32/Video24 locator media -> media-realized attacks",
  "generators": "${GENERATORS}",
  "limit": ${LIMIT},
  "per_generator_limit": ${PER_GENERATOR_LIMIT},
  "splits": "${SPLITS}",
  "selected_clean_manifest": "${SELECTED_CLEAN}",
  "signed_payload_registry": "${SIGNED_PAYLOAD_REGISTRY}",
  "public_key": "${PUBLIC_KEY}",
  "embedded_manifest": "${EMBEDDED_MANIFEST}",
  "rewritten_tasks": "${REWRITTEN_TASKS}",
  "completed_inventory": "${COMPLETED}",
  "output_root": "${OUTPUT_ROOT}",
  "wrapper_config": "${WRAPPER_CONFIG}",
  "wrapper_generators": "${WRAPPER_GENERATORS}",
  "next": "Run feature extraction, M3 fusion, and verify_m6_signed_m4_media_realized.py on this completed inventory."
}
JSON

echo "[m7-m6-e2e] attack pipeline complete -> ${OUT}"
