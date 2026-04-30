#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENES_ROOT="${SCENES_ROOT:-/path/to/generated_scenes}"
OUTPUT_PATH="${OUTPUT_PATH:-${SCRIPT_DIR}/outputs/worldvlm_a1_a4_motion_supervision.jsonl}"
TEMPLATES_PATH="${TEMPLATES_PATH:-${SCRIPT_DIR}/svc_dataset_gen/templates.jsonl}"
SEED="${SEED:-42}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

cd "${SCRIPT_DIR}/svc_dataset_gen"
python generate_undetect_prompts_max_dist.py \
  --data_root "${SCENES_ROOT}" \
  --out_path "${OUTPUT_PATH}" \
  --templates "${TEMPLATES_PATH}" \
  --seed "${SEED}"
