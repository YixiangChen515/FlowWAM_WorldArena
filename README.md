# FlowWAM_WorldArena

FlowWAM for WorldArena evaluation.

## Repository Structure

```
FlowWAM_WorldArena/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── diffsynth/                          # Trimmed diffusion library (Wan + trainers)
├── inference/
│   ├── world_model_inference.sh        # Main entry point
│   ├── world_model_inference.py        # Inference pipeline
│   ├── dataset_world_robotwin.py       # RoboTwin dataset loader
│   ├── reversible_flow_codec.py        # Flow encode / decode
│   ├── video_flow_codec_pipeline.py    # Flow extraction pipeline
│   ├── robot_only_renderer.py          # SAPIEN robot-only renderer
│   ├── generate_summary.py             # Post-inference summary
│   ├── embodiments/                    # Robot URDF configs (download below)
│   ├── refiner/                        # Inline post-processing module
│   └── models/                         # Model checkpoints (download below)
└── training/
    ├── world_model_train.sh            # Training entry point
    ├── world_model_train.py            # Training loop (accelerate + swanlab)
    ├── world_model_module.py           # Dual-stream world model module
    ├── dataset.py                      # RoboTwin training dataset
    ├── sampler.py                      # DDP rollout-aware bucket sampler
    ├── dataset_world_robotwin.py       # Shared RoboTwin helpers
    ├── reversible_flow_codec.py        # Flow encode / decode
    └── video_flow_codec_pipeline.py    # Flow extraction pipeline
```

## Environment Setup

```bash
# 1) Clone repository
git clone https://github.com/YixiangChen515/FlowWAM_WorldArena.git
cd FlowWAM_WorldArena

# 2) Create environment (Python 3.10 + CUDA 12.1 toolchain)
conda create -n flowwam python=3.10 -y
conda activate flowwam

# 3) Install PyTorch (CUDA 12.1 build)
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.0 torchvision==0.18.0

# 4) Build prerequisites + flash-attn + apex (prebuilt wheel)
pip install packaging psutil
pip install flash_attn==2.5.9.post1 --no-build-isolation --no-cache-dir
pip install https://huggingface.co/ByteDance-Seed/SeedVR2-3B/resolve/main/apex-0.1-cp310-cp310-linux_x86_64.whl

# 5) Project dependencies
pip install -r inference/refiner/SeedVR/requirements.txt
pip install -r requirements.txt
pip install -e .

# 6) Pin cuBLAS last to override the version pulled in by transitive deps
pip install nvidia-cublas-cu12==12.4.5.8
```

## Model & Data Download

```bash
pip install huggingface_hub

# 1. Base Wan models
hf download Wan-AI/Wan2.2-TI2V-5B \
    --local-dir inference/models/Wan-AI/Wan2.2-TI2V-5B
hf download Wan-AI/Wan2.1-T2V-1.3B \
    --include "google/*" "models_t5_umt5-xxl-enc-bf16.pth" \
    --local-dir inference/models/Wan-AI/Wan2.1-T2V-1.3B

# 2. FlowWAM checkpoint
hf download YixiangChen/FlowWAM flowwam_worldarena_stage1.safetensors \
    --local-dir inference/models/stage_1/

# 3. Refiner checkpoints
hf download ByteDance-Seed/SeedVR2-3B \
    seedvr2_ema_3b.pth ema_vae.pth \
    --local-dir inference/models/stage_2/

# 4. Robot embodiment configs (~220 MB)
hf download TianxingChen/RoboTwin2.0 embodiments.zip \
    --repo-type dataset --local-dir inference/
cd inference && unzip embodiments.zip && rm embodiments.zip && cd ..
```

## Running Inference

```bash
# Usage: bash inference/world_model_inference.sh <test_dataset_dir> [gpu_ids]
# test_dataset_dir example: /path/to/WorldArena/data/WorldArena_Robotwin2.0/test_dataset

# All available GPUs
bash inference/world_model_inference.sh /path/to/test_dataset

# A specific GPU set
bash inference/world_model_inference.sh /path/to/test_dataset 0,1

# Single GPU
bash inference/world_model_inference.sh /path/to/test_dataset 0
```

The script generates the original-instruction videos and writes
`summary.json` at the end.

## Output Structure

```
inference/FlowWAM_eval/
├── FlowWAM_test/           # Generated videos (original instructions)
│   └── <episode>.mp4
└── summary.json            # Aggregated results
```

## Training

The training code lives in `training/` and is self-contained: it reuses the
same trimmed `diffsynth` library (with `diffsynth/trainers/`) and the Wan base
models downloaded for inference.

### Data layout

Training reads RoboTwin HDF5 episodes from two roots:

- A high-resolution root (`--dataset_base_path`, the `640/` folder below)
  providing supervision frames and robot-only frames (for optical flow).
- A low-resolution root (`--low_res_data_root`, the `320/` folder below)
  providing the conditioning reference frame, which is BICUBIC-upsampled to
  match the inference-time degradation.

Each root follows the standard RoboTwin layout:

```
<data_root>/<task>/<variant>/
├── data/episode*.hdf5                  # scene RGB (observation/<camera>/rgb)
├── robot_only/data/episode*.hdf5       # robot-only RGB (used for flow)
└── instructions/episode*.json          # language instructions (optional)
```

### Download the training data

The RoboTwin training episodes are released on Hugging Face at
[YixiangChen/FlowWAM_WorldArena](https://huggingface.co/datasets/YixiangChen/FlowWAM_WorldArena).
The dataset has two top-level folders that map directly onto the two training
roots:

- `640/` — high-resolution episodes → `--dataset_base_path`
- `320/` — low-resolution episodes → `--low_res_data_root`

Each folder holds one `<task>.zip` per RoboTwin task; every archive extracts to
`<task>/aloha-agilex_clean_50/` with the `data/`, `robot_only/` and
`instructions/` streams the loader expects.

```bash
pip install huggingface_hub

# 1. Download both resolutions (all tasks)
hf download YixiangChen/FlowWAM_WorldArena \
    --repo-type dataset --local-dir data/FlowWAM_WorldArena

# 2. Unpack every task archive in place
cd data/FlowWAM_WorldArena
for f in 640/*.zip; do unzip -q -o "$f" -d 640/; done
for f in 320/*.zip; do unzip -q -o "$f" -d 320/; done
rm 640/*.zip 320/*.zip          # optional: reclaim space
cd -
```

This yields:

```
data/FlowWAM_WorldArena/
├── 640/<task>/aloha-agilex_clean_50/{data,robot_only,instructions}/   # → DATASET_BASE_PATH
└── 320/<task>/aloha-agilex_clean_50/{data,robot_only,instructions}/   # → LOW_RES_DATA_ROOT
```

Then point the training script at the two roots:

```bash
DATASET_BASE_PATH=/abs/path/to/data/FlowWAM_WorldArena/640
LOW_RES_DATA_ROOT=/abs/path/to/data/FlowWAM_WorldArena/320
```

### Models

Training fine-tunes the `Wan2.2-TI2V-5B` backbone. Reuse the base models
downloaded in "Model & Data Download"; `world_model_train.sh` symlinks them
into `training/models/` automatically:

```bash
ln -sfn ../inference/models/Wan-AI training/models/Wan-AI
```

### Launch

Edit `DATASET_BASE_PATH` / `LOW_RES_DATA_ROOT` (and, for multi-node,
`NUM_MACHINES` / `MASTER_ADDR` / `MASTER_PORT`) at the top of
`training/world_model_train.sh`, then:

```bash
cd training

# optional: enable swanlab logging
export SWANLAB_API_KEY=...

# Single node, all visible GPUs
bash world_model_train.sh

# Multi-node: run on every node with matching NUM_MACHINES / MASTER_ADDR
bash world_model_train.sh --machine_rank 0   # master
bash world_model_train.sh --machine_rank 1   # worker 1
```

Checkpoints are written under `OUTPUT_PATH` (default `training/models/train/...`)
every `SAVE_EVERY_N_EPOCHS` epochs. To resume, set `RESUME_CHECKPOINT` to a
saved `epoch-N.safetensors`.

## Third-party components

`inference/refiner/` contains vendored source code distributed under the
Apache-2.0 license. See `inference/refiner/LICENSE_SeedVR` for the full
license text and `inference/refiner/NOTICE` for attribution and a list
of modifications.