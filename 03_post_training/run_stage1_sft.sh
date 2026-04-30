#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-VL-7B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-qwen2_5_vl}"
DATA_PATH="${DATA_PATH:-/path/to/worldvlm_sft_data.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/sft}"

EPOCH="${EPOCH:-1}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LORA_RANK="${LORA_RANK:-256}"
LORA_ALPHA="${LORA_ALPHA:-512}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.95}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
SAVE_STEPS="${SAVE_STEPS:-40}"
EVAL_STEPS="${EVAL_STEPS:-40}"
MAX_PIXELS="${MAX_PIXELS:-1003520}"

CONDA_SH="${CONDA_SH:-/path/to/miniconda3/etc/profile.d/conda.sh}"
SWIFT_ENV="${SWIFT_ENV:-swift}"
LMMS_ENV="${LMMS_ENV:-lmms-eval}"
MINDCUBE_ENV="${MINDCUBE_ENV:-mindcube}"
SAT_ENV="${SAT_ENV:-mindjourney}"
MINDCUBE_ROOT="${MINDCUBE_ROOT:-/path/to/mindcube}"
SAT_ROOT="${SAT_ROOT:-/path/to/mindjourney}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-1}"
SKIP_VSIBENCH="${SKIP_VSIBENCH:-1}"
SKIP_MINDCUBE="${SKIP_MINDCUBE:-1}"
SKIP_SAT="${SKIP_SAT:-1}"
SKIP_SAT_SYNTHESIZED="${SKIP_SAT_SYNTHESIZED:-1}"
ONLY_EVAL_LAST_CKPT="${ONLY_EVAL_LAST_CKPT:-0}"
SLEEP_AFTER_TRAIN="${SLEEP_AFTER_TRAIN:-10}"

CONDA_READY=0

is_true() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

format_cmd() {
  local out=
  local arg
  for arg in "$@"; do
    out+="$(printf '%q' "${arg}") "
  done
  printf '%s' "${out% }"
}

preview_cmd() {
  echo "+ $(format_cmd "$@")" >&2
}

run_cmd() {
  preview_cmd "$@"
  if ! is_true "${DRY_RUN}"; then
    "$@"
  fi
}

run_cmd_in_dir() {
  local workdir=$1
  shift
  echo "+ (cd $(printf '%q' "${workdir}") && $(format_cmd "$@"))" >&2
  if ! is_true "${DRY_RUN}"; then
    (
      cd "${workdir}"
      "$@"
    )
  fi
}

activate_env() {
  local env_name=$1
  if is_true "${DRY_RUN}"; then
    echo "+ conda activate ${env_name}" >&2
    return 0
  fi
  if [[ "${CONDA_READY}" == 0 ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_SH}"
    CONDA_READY=1
  fi
  conda activate "${env_name}"
}

find_latest_run_dir() {
  find "${OUTPUT_DIR}" -maxdepth 1 -mindepth 1 -type d \
    -regextype posix-extended \
    -regex '.*/v[0-9]+-.*' | sort -rV | head -n 1
}

list_checkpoints() {
  local run_dir=$1
  if is_true "${ONLY_EVAL_LAST_CKPT}"; then
    find "${run_dir}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' ! -name '*-merged' | sort -V | tail -n 1
  else
    find "${run_dir}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' ! -name '*-merged' | sort -V
  fi
}

merged_dir_for_checkpoint() {
  printf '%s\n' "${1}-merged"
}

merge_checkpoint() {
  local checkpoint_dir=$1
  local merged_dir

  merged_dir=$(merged_dir_for_checkpoint "${checkpoint_dir}")
  if [[ -d "${merged_dir}" ]]; then
    return 0
  fi

  activate_env "${SWIFT_ENV}"
  run_cmd swift export --adapters "${checkpoint_dir}" --merge_lora true

  if ! is_true "${DRY_RUN}" && [[ ! -d "${merged_dir}" ]]; then
    echo "Expected merged checkpoint not found: ${merged_dir}" >&2
    exit 1
  fi
}

run_vsibench_eval() {
  local model_path=$1
  local run_tag=$2
  local run_dir=$3

  mkdir -p "${run_dir}/eval/lmms/${run_tag}/vsibench"
  activate_env "${LMMS_ENV}"
  run_cmd \
    env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    python -m accelerate.commands.launch \
    --num_processes="${NPROC_PER_NODE}" \
    --main_process_port 0 \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args "pretrained=${model_path}" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix vsibench \
    --output_path "${run_dir}/eval/lmms/${run_tag}/vsibench"
}

run_mindcube_eval() {
  local model_path=$1
  local run_tag=$2
  local run_dir=$3

  mkdir -p "${run_dir}/eval/mindcube/results" "${run_dir}/eval/mindcube/evaluate"
  activate_env "${MINDCUBE_ENV}"
  run_cmd_in_dir "${MINDCUBE_ROOT}" \
    env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    python run_eval_pipeline.py \
    --input-path data/prompts/general/MindCube_tinybench_raw_qa.jsonl \
    --input-kind prompt \
    --model-type qwen2.5vl \
    --model-path "${model_path}" \
    --backend transformers \
    --image-root data \
    --batch-size 1 \
    --chunk-size 128 \
    --gpu-ids "${CUDA_VISIBLE_DEVICES}" \
    --jobs-per-gpu 1 \
    --inference-output-dir "${run_dir}/eval/mindcube/results/${run_tag}" \
    --eval-output-dir "${run_dir}/eval/mindcube/evaluate/${run_tag}" \
    --eval-workers 2 \
    --task cogmap \
    --verbose
}

run_sat_eval() {
  local model_path=$1
  local run_tag=$2
  local run_dir=$3

  mkdir -p "${run_dir}/eval/sat/${run_tag}"
  activate_env "${SAT_ENV}"
  run_cmd_in_dir "${SAT_ROOT}" \
    env WORLD_MODEL_TYPE=svc PYTHONPATH="${PYTHONPATH:-}:./" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    python pipelines/pipeline_qwenvl_base.py \
    --vlm_model_name qwen2.5-vl \
    --local_model_path "${model_path}" \
    --vlm_qa_model_name None \
    --num_questions 150 \
    --output_dir "${run_dir}/eval/sat/${run_tag}" \
    --input_dir data \
    --question_type None \
    --max_images 2 \
    --max_tries_gpt 5 \
    --split test \
    --num_question_chunks 1 \
    --question_chunk_idx 0 \
    --task img2trajvid_s-prob \
    --replace_or_include_input True \
    --cfg 4.0 \
    --guider 1 \
    --L_short 576 \
    --num_targets 8 \
    --use_traj_prior True \
    --chunk_strategy interp
}

run_sat_synthesized_eval() {
  local model_path=$1
  local run_tag=$2
  local run_dir=$3

  mkdir -p "${run_dir}/eval/sat_synthesized/${run_tag}"
  activate_env "${SAT_ENV}"
  run_cmd_in_dir "${SAT_ROOT}" \
    env WORLD_MODEL_TYPE=svc PYTHONPATH="${PYTHONPATH:-}:./" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    python pipelines/pipeline_qwenvl_base.py \
    --vlm_model_name qwen2.5-vl \
    --local_model_path "${model_path}" \
    --vlm_qa_model_name None \
    --num_questions 500 \
    --output_dir "${run_dir}/eval/sat_synthesized/${run_tag}" \
    --input_dir data \
    --question_type None \
    --max_images 2 \
    --max_tries_gpt 5 \
    --split val \
    --num_question_chunks 1 \
    --question_chunk_idx 0 \
    --task img2trajvid_s-prob \
    --replace_or_include_input True \
    --cfg 4.0 \
    --guider 1 \
    --L_short 576 \
    --num_targets 8 \
    --use_traj_prior True \
    --chunk_strategy interp
}

run_all_evals() {
  local run_dir=$1
  local checkpoint_dir
  local merged_dir
  local run_tag

  while IFS= read -r checkpoint_dir; do
    [[ -z "${checkpoint_dir}" ]] && continue
    merge_checkpoint "${checkpoint_dir}"
    merged_dir=$(merged_dir_for_checkpoint "${checkpoint_dir}")
    run_tag="$(basename "${run_dir}")_$(basename "${checkpoint_dir}")"

    if ! is_true "${SKIP_VSIBENCH}"; then
      run_vsibench_eval "${merged_dir}" "${run_tag}" "${run_dir}"
    fi
    if ! is_true "${SKIP_MINDCUBE}"; then
      run_mindcube_eval "${merged_dir}" "${run_tag}" "${run_dir}"
    fi
    if ! is_true "${SKIP_SAT}"; then
      run_sat_eval "${merged_dir}" "${run_tag}" "${run_dir}"
    fi
    if ! is_true "${SKIP_SAT_SYNTHESIZED}"; then
      run_sat_synthesized_eval "${merged_dir}" "${run_tag}" "${run_dir}"
    fi
  done < <(list_checkpoints "${run_dir}")
}

main() {
  local -a train_cmd=(
    env
    NPROC_PER_NODE="${NPROC_PER_NODE}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    MAX_PIXELS="${MAX_PIXELS}"
    swift sft
    --model "${MODEL_PATH}"
    --model_type "${MODEL_TYPE}"
    --dataset "${DATA_PATH}"
    --load_from_cache_file true
    --split_dataset_ratio 0.01
    --dataset_shuffle true
    --train_dataloader_shuffle true
    --train_type lora
    --torch_dtype bfloat16
    --num_train_epochs "${EPOCH}"
    --per_device_train_batch_size "${BATCH_SIZE}"
    --per_device_eval_batch_size 1
    --learning_rate "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --adam_beta1 "${ADAM_BETA1}"
    --adam_beta2 "${ADAM_BETA2}"
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
    --freeze_vit true
    --target_modules all-linear
    --lora_rank "${LORA_RANK}"
    --lora_alpha "${LORA_ALPHA}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --eval_steps "${EVAL_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --logging_steps 5
    --max_length 8192
    --output_dir "${OUTPUT_DIR}"
    --warmup_ratio 0.05
    --dataloader_num_workers 4
    --dataset_num_proc 4
    --deepspeed zero2
    --save_only_model true
    --report_to tensorboard
  )
  local run_dir

  echo "MODEL_PATH=${MODEL_PATH}"
  echo "DATA_PATH=${DATA_PATH}"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"
  echo "SKIP_TRAIN=${SKIP_TRAIN}"
  echo "SKIP_VSIBENCH=${SKIP_VSIBENCH}"
  echo "SKIP_MINDCUBE=${SKIP_MINDCUBE}"
  echo "SKIP_SAT=${SKIP_SAT}"
  echo "SKIP_SAT_SYNTHESIZED=${SKIP_SAT_SYNTHESIZED}"

  activate_env "${SWIFT_ENV}"
  if is_true "${SKIP_TRAIN}"; then
    preview_cmd "${train_cmd[@]}"
  else
    run_cmd "${train_cmd[@]}"
  fi

  if ! is_true "${DRY_RUN}"; then
    sleep "${SLEEP_AFTER_TRAIN}"
  fi

  run_dir=$(find_latest_run_dir)
  if [[ -z "${run_dir}" ]]; then
    echo "No run directory found under ${OUTPUT_DIR}" >&2
    exit 1
  fi

  run_all_evals "${run_dir}"
}

main "$@"
