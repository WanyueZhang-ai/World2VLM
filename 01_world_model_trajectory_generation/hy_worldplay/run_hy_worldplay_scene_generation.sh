#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROMPT="${PROMPT:-A paved pathway leads towards a stone arch bridge spanning a calm body of water. Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky. The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere.}"
INPUT_DIR="${INPUT_DIR:-/path/to/simulate_rgb}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs}"
VERSION="${VERSION:-simulate_worldplay_demo}"
MOTION_CONFIG="${MOTION_CONFIG:-${SCRIPT_DIR}/motion_configs/simulate_datagen.yaml}"

MODEL_PATH="${MODEL_PATH:-/path/to/hunyuanvideo_1_5}"
ACTION_MODEL_PATH="${ACTION_MODEL_PATH:-/path/to/worldplay_action_model.safetensors}"

N_INFERENCE_GPU="${N_INFERENCE_GPU:-8}"
SEED="${SEED:-1}"
SAVE_MP4="${SAVE_MP4:-false}"
MAX_IMAGES="${MAX_IMAGES:-}"

RESOLUTION="${RESOLUTION:-480p}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"
FEW_STEP="${FEW_STEP:-true}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-4}"
MODEL_TYPE="${MODEL_TYPE:-ar}"
REWRITE="${REWRITE:-false}"
ENABLE_SR="${ENABLE_SR:-false}"
SAVE_PRE_SR_VIDEO="${SAVE_PRE_SR_VIDEO:-false}"
OFFLOADING="${OFFLOADING:-true}"
GROUP_OFFLOADING="${GROUP_OFFLOADING:-}"
DTYPE="${DTYPE:-bf16}"
WIDTH="${WIDTH:-832}"
HEIGHT="${HEIGHT:-480}"

ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-false}"
USE_SAGEATTN="${USE_SAGEATTN:-false}"
SAGE_BLOCKS_RANGE="${SAGE_BLOCKS_RANGE:-0-53}"
USE_VAE_PARALLEL="${USE_VAE_PARALLEL:-false}"
USE_FP8_GEMM="${USE_FP8_GEMM:-false}"
QUANT_TYPE="${QUANT_TYPE:-fp8-per-block}"
INCLUDE_PATTERNS="${INCLUDE_PATTERNS:-double_blocks}"

ARGS=(
  --input_dir "${INPUT_DIR}"
  --output_root "${OUTPUT_ROOT}"
  --version "${VERSION}"
  --prompt "${PROMPT}"
  --seed "${SEED}"
  --skip_existing
  --save_mp4 "${SAVE_MP4}"
  --motion_config "${MOTION_CONFIG}"
  --model_path "${MODEL_PATH}"
  --action_ckpt "${ACTION_MODEL_PATH}"
  --resolution "${RESOLUTION}"
  --aspect_ratio "${ASPECT_RATIO}"
  --few_step "${FEW_STEP}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --model_type "${MODEL_TYPE}"
  --rewrite "${REWRITE}"
  --sr "${ENABLE_SR}"
  --save_pre_sr_video "${SAVE_PRE_SR_VIDEO}"
  --offloading "${OFFLOADING}"
  --dtype "${DTYPE}"
  --enable_torch_compile "${ENABLE_TORCH_COMPILE}"
  --use_sageattn "${USE_SAGEATTN}"
  --sage_blocks_range "${SAGE_BLOCKS_RANGE}"
  --use_vae_parallel "${USE_VAE_PARALLEL}"
  --use_fp8_gemm "${USE_FP8_GEMM}"
  --quant_type "${QUANT_TYPE}"
  --include_patterns "${INCLUDE_PATTERNS}"
)

if [[ -n "${MAX_IMAGES}" ]]; then
  ARGS+=(--max_images "${MAX_IMAGES}")
fi

if [[ -n "${GROUP_OFFLOADING}" ]]; then
  ARGS+=(--group_offloading "${GROUP_OFFLOADING}")
fi

if [[ -n "${HEIGHT}" ]]; then
  ARGS+=(--height "${HEIGHT}")
fi

if [[ -n "${WIDTH}" ]]; then
  ARGS+=(--width "${WIDTH}")
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

cd "${SCRIPT_DIR}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${N_INFERENCE_GPU}" \
  --module hyvideo.batch_generate \
  "${ARGS[@]}"
