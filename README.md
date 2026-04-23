# FlowWAM_WorldArena

FlowWAM for WorldArena evaluation.

## Repository Structure

```
FlowWAM_WorldArena/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── diffsynth/                          # Trimmed diffusion library (Wan-only)
└── inference/
    ├── world_model_inference.sh        # Main entry point
    ├── world_model_inference.py        # Inference pipeline
    ├── dataset_world_robotwin.py       # RoboTwin dataset loader
    ├── reversible_flow_codec.py        # Flow encode / decode
    ├── video_flow_codec_pipeline.py    # Flow extraction pipeline
    ├── robot_only_renderer.py          # SAPIEN robot-only renderer
    ├── generate_summary.py             # Post-inference summary
    ├── action_triplets.json            # Cross-episode action mapping
    ├── embodiments/                    # Robot URDF configs (download below)
    ├── refiner/                        # Inline post-processing module
    └── models/                         # Model checkpoints (download below)
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
hf download YixiangChen/FlowWAM flowwam_stage1.safetensors \
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

The script runs all three instruction variants (original + two
cross-action) and writes `summary.json` at the end.

## Output Structure

```
inference/FlowWAM_eval/
├── FlowWAM_test/           # Variant 0 (original instructions)
│   └── <episode>.mp4
├── FlowWAM_test_1/         # Variant 1 (cross-action)
├── FlowWAM_test_2/         # Variant 2 (cross-action)
└── summary.json            # Aggregated results
```

## Third-party components

`inference/refiner/` contains vendored source code distributed under the
Apache-2.0 license. See `inference/refiner/LICENSE_SeedVR` for the full
license text and `inference/refiner/NOTICE` for attribution and a list
of modifications.