#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENES_ROOT="${SCENES_ROOT:-/path/to/generated_scenes}"
OUTPUT_PATH="${OUTPUT_PATH:-${SCRIPT_DIR}/outputs/worldvlm_d1_d4_object_supervision.jsonl}"
DETECT_MODEL="${DETECT_MODEL:-/path/to/yolov8l-worldv2.pt}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda:0}"
CONF="${CONF:-0.3}"
IOU="${IOU:-0.5}"
MIN_AREA="${MIN_AREA:-0.01}"
MAX_AREA="${MAX_AREA:-0.6}"
EDGE_MARGIN="${EDGE_MARGIN:-0.01}"
MAX_SCENES="${MAX_SCENES:-2000}"
CONF_KEEP="${CONF_KEEP:-0.3}"

export CUDA_VISIBLE_DEVICES
mkdir -p "$(dirname "${OUTPUT_PATH}")"

cd "${SCRIPT_DIR}/svc_dataset_gen"
python generate_detect_prompts_norm_1000_max_dist.py \
  --scenes-root "${SCENES_ROOT}" \
  --output "${OUTPUT_PATH}" \
  --model "${DETECT_MODEL}" \
  --device "${DEVICE}" \
  --conf "${CONF}" \
  --iou "${IOU}" \
  --min-area "${MIN_AREA}" \
  --max-area "${MAX_AREA}" \
  --edge-margin "${EDGE_MARGIN}" \
  --max-scenes "${MAX_SCENES}" \
  --conf-keep "${CONF_KEEP}" \
  --verbose
