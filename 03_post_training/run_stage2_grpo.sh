#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPPORT_DIR="${SCRIPT_DIR}/grpo_support"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_COUNT="${GPU_COUNT:-4}"
MODEL_PATH="${MODEL_PATH:-/path/to/merged_sft_checkpoint}"
RAW_DATA_PATH="${RAW_DATA_PATH:-/path/to/worldvlm_grpo_data.jsonl}"
DATA_PREP_SCRIPT="${DATA_PREP_SCRIPT:-${SUPPORT_DIR}/prepare_grpo_data.py}"
REWARD_FUNC="${REWARD_FUNC:-${SCRIPT_DIR}/reward/worldvlm_reward.py:compute_score}"
GRPO_CONFIG="${GRPO_CONFIG:-${SUPPORT_DIR}/config.yaml}"
PROJECT_NAME="${PROJECT_NAME:-worldvlm_grpo}"
TRAIN_TAG="${TRAIN_TAG:-worldvlm_multitype_coldstart}"
WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/outputs}"
DATA_DIR="${DATA_DIR:-${WORK_DIR}/prepared_data/${TRAIN_TAG}}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${DATA_DIR}/train.jsonl}"
VAL_DATA_PATH="${VAL_DATA_PATH:-${DATA_DIR}/val.jsonl}"
VAL_RATIO="${VAL_RATIO:-0.01}"
SEED="${SEED:-42}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-${GPU_COUNT}}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
ROLLOUT_N="${ROLLOUT_N:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
LIMIT_IMAGES="${LIMIT_IMAGES:-2}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"

mkdir -p "${DATA_DIR}"

if [[ ! -f "${TRAIN_DATA_PATH}" || ! -f "${VAL_DATA_PATH}" ]]; then
  python3 "${DATA_PREP_SCRIPT}" \
    --input "${RAW_DATA_PATH}" \
    --train_output "${TRAIN_DATA_PATH}" \
    --val_output "${VAL_DATA_PATH}" \
    --val_ratio "${VAL_RATIO}" \
    --seed "${SEED}"
fi

python3 -m verl.trainer.main \
  config="${GRPO_CONFIG}" \
  data.train_files="${TRAIN_DATA_PATH}" \
  data.val_files="${VAL_DATA_PATH}" \
  data.prompt_key=prompt \
  data.answer_key=answer \
  data.image_key=images \
  data.max_prompt_length=1024 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.rollout_batch_size="${ROLLOUT_BATCH_SIZE}" \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.actor.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  worker.actor.micro_batch_size_per_device_for_update="${MICRO_BATCH_SIZE}" \
  worker.actor.micro_batch_size_per_device_for_experience="${MICRO_BATCH_SIZE}" \
  worker.critic.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  worker.critic.micro_batch_size_per_device_for_update="${MICRO_BATCH_SIZE}" \
  worker.critic.micro_batch_size_per_device_for_experience="${MICRO_BATCH_SIZE}" \
  worker.rollout.n="${ROLLOUT_N}" \
  worker.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  worker.rollout.limit_images="${LIMIT_IMAGES}" \
  worker.ref.offload.offload_params=true \
  worker.reward.reward_function="${REWARD_FUNC}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="qwen2_5_vl_7b_grpo_${TRAIN_TAG}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.n_gpus_per_node="${GPU_COUNT}"
