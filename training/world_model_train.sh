#!/bin/bash
# Dual-stream world model training on RoboTwin (single- or multi-node).
#
# Usage:
#   bash world_model_train.sh                    # single node, all visible GPUs
#   bash world_model_train.sh --machine_rank 0   # multi-node: master
#   bash world_model_train.sh --machine_rank 1   # multi-node: worker 1
#
# Before running:
#   - Activate your Python environment (e.g. `conda activate <env>`).
#   - `export SWANLAB_API_KEY=...` if you want swanlab logging.
#   - Set DATASET_BASE_PATH / LOW_RES_DATA_ROOT below to your data roots.
#   - For multi-node, set NUM_MACHINES / MASTER_ADDR / MASTER_PORT below.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Reuse the Wan base models downloaded for inference (see README "Model Download").
mkdir -p "${SCRIPT_DIR}/models"
ln -sfn "${SCRIPT_DIR}/../inference/models/Wan-AI" "${SCRIPT_DIR}/models/Wan-AI" 2>/dev/null

# ============================================================
#  Distributed config
# ============================================================
NUM_MACHINES=1
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500
MACHINE_RANK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --machine_rank)
      if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: --machine_rank requires a value"
        exit 1
      fi
      MACHINE_RANK="$2"
      shift 2
      ;;
    *)
      echo "Error: unknown argument: $1"
      echo "Supported arguments: --machine_rank <int>"
      exit 1
      ;;
  esac
done

NUM_GPUS=$(nvidia-smi -L | wc -l)
export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((NUM_GPUS - 1)))

if [ "${NUM_MACHINES}" -gt 1 ]; then
  # Force NCCL into plain TCP-over-ethernet mode. On clusters without an
  # IB/RDMA fabric, leaving GPUDirect RDMA on (or letting ranks pick different
  # NICs) makes DDP init fail with "Connect ... retcode 3". Pinning the socket
  # interfaces + disabling IB/GDR keeps cross-node proxy connects on one route.
  export NCCL_SOCKET_IFNAME=eth0,eth1,eth2,eth3
  export NCCL_IB_DISABLE=1
  export NCCL_NET_GDR_LEVEL=0
  export NCCL_CROSS_NIC=0
  export NCCL_DEBUG=INFO
  export NCCL_SOCKET_NTHREADS=4
  export NCCL_NSOCKS_PERTHREAD=4
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
  export NCCL_TIMEOUT=1800000
  ulimit -n 65536 2>/dev/null

  if [ "${MACHINE_RANK}" != "0" ]; then
    echo "[NCCL] Testing connectivity to master ${MASTER_ADDR}:${MASTER_PORT} ..."
    timeout 10 bash -c "echo > /dev/tcp/${MASTER_ADDR}/${MASTER_PORT}" 2>/dev/null \
      && echo "[NCCL] Master reachable." \
      || echo "[WARN] Cannot reach master at ${MASTER_ADDR}:${MASTER_PORT}!"
  fi
fi

# ============================================================
#  Training hyper-parameters
# ============================================================
DATASET_BASE_PATH="/path/to/robotwin/dataset_highres"   # hi-res supervision + robot-only
LOW_RES_DATA_ROOT="/path/to/robotwin/dataset_lowres"    # lo-res conditioning frame

MODEL_PATHS="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"

# VARIANT: the variant subdir name under each task directory
# (<data_root>/<task>/<variant>/). The released FlowWAM_WorldArena dataset
# ships a single clean variant per task.
VARIANT=("aloha-agilex_clean_50")
CAMERA="head_camera"

NUM_FRAMES=121
SIZE_W=640
SIZE_H=480
LEARNING_RATE=5e-5
NUM_EPOCHS=500

MAX_STRIDE=3
MAX_ROLLOUTS=2
BUCKET_OVERSAMPLE=2

FLOW_METHOD="raft"
FLOW_DEVICE="cuda"
FLOW_MAX_MAGNITUDE=20.0

# ---------- Training mode ----------
TRAINABLE_MODELS="dit"

BATCH_SIZE=1
FLOW_LOSS_WEIGHT=0.0
GRADIENT_ACCUMULATION_STEPS=1
REF_AUG_STRENGTH=0.1
SAVE_EVERY_N_EPOCHS=20
FP32_MODULATION=true

if [ "${FLOW_MAX_MAGNITUDE}" = "-1" ]; then
  FLOW_NORM_TAG="diag"
elif [ -z "${FLOW_MAX_MAGNITUDE}" ]; then
  FLOW_NORM_TAG="auto"
else
  FLOW_NORM_TAG="mag${FLOW_MAX_MAGNITUDE}"
fi
OUTPUT_PATH="./models/train/robotwin_dual_stream_${FLOW_NORM_TAG}"
DATASET_NUM_WORKERS=0

RESUME_CHECKPOINT=""

echo "========== RoboTwin Dual-Stream World Model Training =========="
echo "  NUM_MACHINES:         ${NUM_MACHINES}"
if [ "${NUM_MACHINES}" -gt 1 ]; then
echo "  MASTER_ADDR:          ${MASTER_ADDR}:${MASTER_PORT}"
echo "  MACHINE_RANK:         ${MACHINE_RANK}"
fi
echo "  GPUs per node:        ${NUM_GPUS} (${CUDA_VISIBLE_DEVICES})"
echo "  Total GPUs:           $((NUM_GPUS * NUM_MACHINES))"
echo "  dataset (hi-res):     ${DATASET_BASE_PATH}"
echo "  dataset (lo-res ref): ${LOW_RES_DATA_ROOT}"
echo "  variant:              ${VARIANT[*]}"
echo "  camera:               ${CAMERA}"
echo "  num_frames:           ${NUM_FRAMES}"
echo "  size:                 ${SIZE_W}x${SIZE_H}"
echo "  max_stride:           ${MAX_STRIDE}"
echo "  max_rollouts:         ${MAX_ROLLOUTS}"
echo "  bucket_oversample:    ${BUCKET_OVERSAMPLE}"
echo "  learning_rate:        ${LEARNING_RATE}"
echo "  num_epochs:           ${NUM_EPOCHS}"
echo "  flow_method:          ${FLOW_METHOD}"
echo "  flow_max_magnitude:   ${FLOW_MAX_MAGNITUDE}"
echo "  batch_size:           ${BATCH_SIZE}"
echo "  flow_loss_weight:     ${FLOW_LOSS_WEIGHT}"
echo "  gradient_accum:       ${GRADIENT_ACCUMULATION_STEPS}"
echo "  ref_aug_strength:     ${REF_AUG_STRENGTH}"
echo "  save_every_n_epochs:  ${SAVE_EVERY_N_EPOCHS}"
echo "  fp32_modulation:      ${FP32_MODULATION:-false}"
echo "  output_path:          ${OUTPUT_PATH}"
echo "  resume_checkpoint:    ${RESUME_CHECKPOINT:-<none>}"
echo "==============================================================="

if [[ -z "${LORA_BASE_MODEL}" && -z "${TRAINABLE_MODELS}" ]]; then
  echo "ERROR: Must set one of LORA_BASE_MODEL or TRAINABLE_MODELS."
  exit 1
fi

TRAINING_MODE_ARGS=""
if [[ -n "${LORA_BASE_MODEL}" ]]; then
  echo "  mode:                 LoRA (base=${LORA_BASE_MODEL}, rank=${LORA_RANK})"
  TRAINING_MODE_ARGS="--lora_base_model ${LORA_BASE_MODEL} --lora_target_modules ${LORA_TARGET_MODULES} --lora_rank ${LORA_RANK}"
else
  echo "  mode:                 Full fine-tuning (trainable=${TRAINABLE_MODELS})"
  TRAINING_MODE_ARGS="--trainable_models ${TRAINABLE_MODELS}"
fi

ACCELERATE_ARGS="--num_processes=$((NUM_GPUS * NUM_MACHINES)) --num_machines=${NUM_MACHINES}"
if [ "${NUM_MACHINES}" -gt 1 ]; then
  ACCELERATE_ARGS="${ACCELERATE_ARGS} --machine_rank=${MACHINE_RANK} --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT}"
  if [ "${MACHINE_RANK}" = "0" ]; then
    echo "[Multi-Node] MASTER node (rank 0), waiting for ${NUM_MACHINES} nodes on ${MASTER_ADDR}:${MASTER_PORT} ..."
  else
    echo "[Multi-Node] WORKER node (rank ${MACHINE_RANK}), connecting to ${MASTER_ADDR}:${MASTER_PORT} ..."
  fi
else
  echo "[Single-Node] Launching on ${NUM_GPUS} GPUs."
fi

accelerate launch \
  ${ACCELERATE_ARGS} \
  "${SCRIPT_DIR}/world_model_train.py" \
  --dataset_base_path ${DATASET_BASE_PATH} \
  --low_res_data_root ${LOW_RES_DATA_ROOT} \
  --num_frames ${NUM_FRAMES} \
  --model_id_with_origin_paths "${MODEL_PATHS}" \
  --learning_rate ${LEARNING_RATE} \
  --num_epochs ${NUM_EPOCHS} \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --size ${SIZE_W} ${SIZE_H} \
  --variant "${VARIANT[@]}" \
  --camera ${CAMERA} \
  ${TRAINING_MODE_ARGS} \
  --flow_method ${FLOW_METHOD} \
  --flow_device ${FLOW_DEVICE} \
  --flow_max_magnitude ${FLOW_MAX_MAGNITUDE} \
  --dataset_num_workers ${DATASET_NUM_WORKERS} \
  --flow_loss_weight ${FLOW_LOSS_WEIGHT} \
  --batch_size ${BATCH_SIZE} \
  --find_unused_parameters \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --ref_aug_strength ${REF_AUG_STRENGTH} \
  --save_every_n_epochs ${SAVE_EVERY_N_EPOCHS} \
  --max_stride ${MAX_STRIDE} \
  --max_rollouts ${MAX_ROLLOUTS} \
  --bucket_oversample ${BUCKET_OVERSAMPLE} \
  ${FP32_MODULATION:+--fp32_modulation} \
  --use_gradient_checkpointing \
  ${RESUME_CHECKPOINT:+--resume_checkpoint "${RESUME_CHECKPOINT}"}
