#!/bin/bash
# FlowWAM world model inference for WorldArena evaluation.
# Runs 3 instruction variants (0, 1, 2) and generates summary.json.
#
# Usage:
#   bash world_model_inference.sh <test_dataset_dir> [gpu_ids]
#
# Examples:
#   bash world_model_inference.sh /data/test_dataset          # all GPUs
#   bash world_model_inference.sh /data/test_dataset 0,1      # GPU 0 and 1
#   bash world_model_inference.sh /data/test_dataset 0        # GPU 0 only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- Parse positional arguments ----------
if [[ -z "$1" ]]; then
  echo "Usage: bash world_model_inference.sh <test_dataset_dir> [gpu_ids]"
  echo "  gpu_ids: comma-separated GPU indices, e.g. 0,1,2  (default: all GPUs)"
  exit 1
fi

TEST_DATASET_DIR="$1"

if [[ -n "$2" ]]; then
  export CUDA_VISIBLE_DEVICES="$2"
  NUM_GPUS=$(echo "$2" | tr ',' '\n' | wc -l)
else
  NUM_GPUS=$(nvidia-smi -L | wc -l)
  export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((NUM_GPUS - 1)))
fi

export MASTER_PORT=29500

export CUDNN_LOGINFO_DBG=0
export CUDNN_LOGERR_DBG=0
export PYTHONWARNINGS="ignore::UserWarning"

# ============================================================
#  Inference configuration
# ============================================================
MODEL_NAME="FlowWAM"
FULL_PATH="${SCRIPT_DIR}/models/stage_1/flowwam_stage1.safetensors"

# embodiments/ should be in this directory (see README for download instructions)
EMBODIMENT_DIR="${SCRIPT_DIR}"

VARIANT="aloha-agilex_clean_50"
OUTPUT_DIR="${SCRIPT_DIR}/${MODEL_NAME}_eval"

MAX_STRIDE=3
MAX_ROLLOUTS=2

NUM_OUTPUT_FRAMES=121
SIZE_W=640
SIZE_H=480
FPS=24
CAMERA="head_camera"

FLOW_METHOD="raft"
FLOW_DEVICE="cuda"
FLOW_MAX_MAGNITUDE=20.0
FLOW_RES_W=320
FLOW_RES_H=240

RENDER_RES_W=640
RENDER_RES_H=480

NUM_INFERENCE_STEPS=50
SIGMA_SHIFT=5.0
SEED=1
NUM_WORKERS=0

MAX_EPISODES=""
EPISODES=""

echo "========== FlowWAM Inference (WorldArena) =========="
echo "  GPUs:                  ${NUM_GPUS} (${CUDA_VISIBLE_DEVICES})"
echo "  test_dataset_dir:      ${TEST_DATASET_DIR}"
echo "  embodiment_dir:        ${EMBODIMENT_DIR}"
echo "  variant:               ${VARIANT}"
echo "  output_dir:            ${OUTPUT_DIR}"
echo "  model_name:            ${MODEL_NAME}"
echo "  checkpoint:            ${FULL_PATH}"
echo "  num_output_frames:     ${NUM_OUTPUT_FRAMES}"
echo "  fps:                   ${FPS}"
echo "  render_resolution:     ${RENDER_RES_W}x${RENDER_RES_H}"
echo "  num_inference_steps:   ${NUM_INFERENCE_STEPS}"
echo "====================================================================="

if [[ ! -f "${FULL_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${FULL_PATH}"
  exit 1
fi

TRIPLET_JSON="${SCRIPT_DIR}/action_triplets.json"

for INSTR_VARIANT in 0 1 2; do
  INSTR_LABEL="instructions"
  [[ ${INSTR_VARIANT} -gt 0 ]] && INSTR_LABEL="instructions_${INSTR_VARIANT}"
  DIR_SUFFIX="${MODEL_NAME}_test"
  [[ ${INSTR_VARIANT} -gt 0 ]] && DIR_SUFFIX="${MODEL_NAME}_test_${INSTR_VARIANT}"

  CROSS_ACTION_ARGS=""
  if [[ ${INSTR_VARIANT} -gt 0 ]]; then
    if [[ -f "${TRIPLET_JSON}" ]]; then
      CROSS_ACTION_ARGS="--triplet_json ${TRIPLET_JSON} --triplet_variant ${INSTR_VARIANT}"
    else
      CROSS_ACTION_ARGS="--cross_episode_action --action_shuffle_seed $((42 + INSTR_VARIANT))"
    fi
  fi

  echo ""
  echo ">>>>>> Action Following variant ${INSTR_VARIANT}: ${INSTR_LABEL} -> ${DIR_SUFFIX}/"
  [[ -n "${CROSS_ACTION_ARGS}" ]] && echo "       ${CROSS_ACTION_ARGS}"
  echo ""

  accelerate launch \
      --num_processes=${NUM_GPUS} \
      --num_machines=1 \
      "${SCRIPT_DIR}/world_model_inference.py" \
      --test_dataset_dir "${TEST_DATASET_DIR}" \
      --embodiment_dir "${EMBODIMENT_DIR}" \
      --variant "${VARIANT}" \
      --output_dir "${OUTPUT_DIR}" \
      --model_name "${MODEL_NAME}" \
      --full_path "${FULL_PATH}" \
      --instruction_variant ${INSTR_VARIANT} \
      --num_output_frames ${NUM_OUTPUT_FRAMES} \
      --size ${SIZE_W} ${SIZE_H} \
      --fps ${FPS} \
      --camera ${CAMERA} \
      --flow_method ${FLOW_METHOD} \
      --flow_device ${FLOW_DEVICE} \
      --flow_max_magnitude ${FLOW_MAX_MAGNITUDE} \
      --flow_resolution ${FLOW_RES_W} ${FLOW_RES_H} \
      --robot_render_resolution ${RENDER_RES_W} ${RENDER_RES_H} \
      --max_stride ${MAX_STRIDE} \
      --max_rollouts ${MAX_ROLLOUTS} \
      --num_inference_steps ${NUM_INFERENCE_STEPS} \
      --sigma_shift ${SIGMA_SHIFT} \
      --seed ${SEED} \
      --num_workers ${NUM_WORKERS} \
      ${MAX_EPISODES:+--max_episodes "${MAX_EPISODES}"} \
      ${EPISODES:+--episodes ${EPISODES}} \
      ${CROSS_ACTION_ARGS}

  echo ""
  echo "<<<<<< Finished variant ${INSTR_VARIANT}"
  echo ""
done

echo "All 3 instruction variants completed."

echo ""
echo ">>>>>> Generating summary.json ..."
python "${SCRIPT_DIR}/generate_summary.py" \
    --eval_dir "${OUTPUT_DIR}" \
    --test_dataset_dir "${TEST_DATASET_DIR}" \
    --model_name "${MODEL_NAME}"

echo ""
echo "===== Inference + Summary complete. Ready for evaluation. ====="
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  Summary:      ${OUTPUT_DIR}/summary.json"
