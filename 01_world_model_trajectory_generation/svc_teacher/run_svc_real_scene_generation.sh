#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SVC_REPO_ROOT="${SVC_REPO_ROOT:-/path/to/stable-virtual-camera}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}"

PYTHONPATH_ENTRIES=("${SCRIPT_DIR}")
if [[ -n "${SVC_REPO_ROOT}" ]]; then
  PYTHONPATH_ENTRIES+=("${SVC_REPO_ROOT}")
fi
if [[ -n "${EXTRA_PYTHONPATH}" ]]; then
  PYTHONPATH_ENTRIES+=("${EXTRA_PYTHONPATH}")
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
  PYTHONPATH_ENTRIES+=("${PYTHONPATH}")
fi
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_ENTRIES[*]}")"

VERSION="${VERSION:-svc_real_demo}"
REAL_CONFIG="${REAL_CONFIG:-${SCRIPT_DIR}/config/svc_teacher_generation.yaml}"
REAL_INPUT="${REAL_INPUT:-/path/to/real_rgb}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_ROOT}/${VERSION}}"
MANIFEST="${MANIFEST:-${RUN_ROOT}/manifest.jsonl}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

SKIP_FLAG=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  SKIP_FLAG=(--skip_existing)
fi

mkdir -p "${RUN_ROOT}"
if [[ ! -s "${MANIFEST}" ]]; then
  mapfile -t IMAGE_FILES < <(
    find "${REAL_INPUT}" -maxdepth 1 -type f \
      \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) \
      | sort
  )

  if [[ "${#IMAGE_FILES[@]}" -eq 0 ]]; then
    echo "No input images found under ${REAL_INPUT}" >&2
    exit 1
  fi

  tmp_manifest="${MANIFEST}.tmp.$$"
  for image_path in "${IMAGE_FILES[@]}"; do
    stem="$(basename "${image_path%.*}")"
    printf '{"scene_id":"%s","anchor_path":"%s"}\n' "${stem}" "$(realpath "${image_path}")"
  done > "${tmp_manifest}"
  mv "${tmp_manifest}" "${MANIFEST}"
fi

IFS=, read -ra GPU_ARR <<< "${GPUS}"
NUM_CHUNKS=${#GPU_ARR[@]}

cd "${SCRIPT_DIR}"

if [[ "${NUM_CHUNKS}" -le 1 ]]; then
  CUDA_VISIBLE_DEVICE_IDX=0
  if [[ "${NUM_CHUNKS}" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICE_IDX="${GPU_ARR[0]}"
  fi
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICE_IDX}" python -m datagen.generate \
    --config "${REAL_CONFIG}" \
    --manifest_jsonl "${MANIFEST}" \
    --out_root "${OUTPUT_ROOT}" \
    --version "${VERSION}" \
    --device cuda:0 \
    "${SKIP_FLAG[@]}"
else
  for i in "${!GPU_ARR[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_ARR[i]}" python -m datagen.generate \
      --config "${REAL_CONFIG}" \
      --manifest_jsonl "${MANIFEST}" \
      --out_root "${OUTPUT_ROOT}" \
      --version "${VERSION}" \
      --num_chunks "${NUM_CHUNKS}" \
      --chunk_idx "${i}" \
      --device cuda:0 \
      "${SKIP_FLAG[@]}" &
  done
  wait
  python -m datagen.generate \
    --config "${REAL_CONFIG}" \
    --out_root "${OUTPUT_ROOT}" \
    --version "${VERSION}" \
    --num_chunks "${NUM_CHUNKS}" \
    --merge_chunks
fi

echo "Scene generation complete: ${RUN_ROOT}/scenes"
