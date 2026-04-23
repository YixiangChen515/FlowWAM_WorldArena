"""
FlowWAM dual-stream world model inference for WorldArena evaluation.
"""

import os
import sys
import math
import json
import random
import logging
import argparse
import datetime
from typing import List, Optional, Tuple

import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_dit_dual_stream import init_flow_stream
from diffsynth.pipelines.wan_video_dual_stream import model_fn_wan_video_dual_stream

import imageio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_world_robotwin import (
    RoboTwinWorldModelInferenceDataset,
    add_bg_texture,
    mask_flows_by_robot,
    _uniform_resample_indices,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  FP32 modulation restore
# ---------------------------------------------------------------------------

def _apply_fp32_modulation(dit, fp32_state_values):
    """Re-apply fp32 values for AdaLN params after VRAM wrapping."""
    from diffsynth.vram_management.layers import AutoWrappedLinear, WanAutoCastLayerNorm

    param_map = dict(dit.named_parameters())
    restored = 0
    for key, fp32_value in fp32_state_values.items():
        if key in param_map:
            param_map[key].data = fp32_value.to(device=param_map[key].device)
            restored += param_map[key].numel()

    for seq_module in [dit.time_embedding, dit.time_projection]:
        for sub in seq_module.modules():
            if isinstance(sub, AutoWrappedLinear):
                sub.offload_dtype = torch.float32
                sub.onload_dtype = torch.float32
                sub.computation_dtype = torch.float32

    def _pre_hook(_mod, args):
        return tuple(a.float() if isinstance(a, torch.Tensor) else a for a in args)

    def _post_hook(_mod, _args, output):
        return output.bfloat16() if isinstance(output, torch.Tensor) else output

    n_hooked = 0
    for seq_module in [dit.time_embedding, dit.time_projection]:
        seq_module.register_forward_pre_hook(_pre_hook)
        seq_module.register_forward_hook(_post_hook)
        n_hooked += 1

    for module in dit.modules():
        if isinstance(module, WanAutoCastLayerNorm):
            module.offload_dtype = torch.float32
            module.onload_dtype = torch.float32
            n_hooked += 1

    _log.info(
        f"[FP32Modulation] Restored {restored:,} fp32 params, "
        f"{n_hooked} modules patched"
    )


# ---------------------------------------------------------------------------
#  Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(
    device: torch.device,
    full_path: str,
    local_model_path: Optional[str] = None,
) -> Tuple[WanVideoPipeline, nn.Module]:
    """Build pipeline + FlowStreamModule from a checkpoint."""
    if local_model_path is None:
        local_model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models"
        )

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=str(device),
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B",
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                        offload_device="cpu",
                        local_model_path=local_model_path),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B",
                        origin_file_pattern="diffusion_pytorch_model*.safetensors",
                        offload_device="cpu",
                        local_model_path=local_model_path),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B",
                        origin_file_pattern="Wan2.2_VAE.pth",
                        offload_device="cpu",
                        local_model_path=local_model_path),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B",
                                     origin_file_pattern="google/*",
                                     local_model_path=local_model_path),
    )

    flow_stream = init_flow_stream(pipe.dit)

    state_dict = load_state_dict(full_path)
    dit_keys = {}
    flow_keys = {}
    for k, v in state_dict.items():
        if k.startswith("flow_stream."):
            flow_keys[k.replace("flow_stream.", "")] = v
        else:
            dit_keys[k] = v

    fp32_dit_values = None
    if dit_keys:
        fp32_dit_values = {k: v.clone() for k, v in dit_keys.items()
                           if v.dtype == torch.float32}
        missing, unexpected = pipe.dit.load_state_dict(dit_keys, strict=False)
        _log.info(
            f"DiT: loaded {len(dit_keys) - len(unexpected)} keys, "
            f"{len(missing)} missing, {len(unexpected)} unexpected, "
            f"{len(fp32_dit_values)} fp32 keys preserved"
        )
    if flow_keys:
        flow_stream.load_state_dict(flow_keys, strict=False)
        _log.info(f"FlowStream: loaded {len(flow_keys)} keys")

    if device.type == "cuda":
        pipe.enable_vram_management()
    else:
        _log.warning(
            "CUDA device not detected; skip enable_vram_management(). "
            "Current device: %s",
            device,
        )

    if fp32_dit_values:
        _apply_fp32_modulation(pipe.dit, fp32_dit_values)

    flow_stream = flow_stream.to(device=device, dtype=torch.bfloat16)
    flow_stream.eval()

    return pipe, flow_stream



# ---------------------------------------------------------------------------
#  Inference dataset with stride-capped rollout
# ---------------------------------------------------------------------------

class RoboTwinRolloutInferenceDataset(RoboTwinWorldModelInferenceDataset):
    """Extends base dataset with multi-chunk rollout support."""

    def __init__(self, *args, max_stride=3, max_rollouts=2, **kwargs):
        self.max_stride = max_stride
        self.max_rollouts = max_rollouts
        super().__init__(*args, **kwargs)

    def _get_robot_only_frames_all(self, sample):
        if sample["robot_only_hdf5"] is not None:
            with h5py.File(sample["robot_only_hdf5"], "r") as f_ro:
                T_ro = f_ro[f"observation/{self.camera}/rgb"].shape[0]
                return [
                    self._load_hdf5_rgb_frame_raw(f_ro, self.camera, i)
                    for i in range(T_ro)
                ]
        else:
            self._ensure_robot_renderer()
            renderer = self._robot_renderer
            T = renderer.get_episode_length(sample["action_hdf5"])
            return renderer.render_episode(
                sample["action_hdf5"], list(range(T)), camera=self.camera
            )

    def _build_flat_indices(self, T, num_rollouts):
        chunk_output_frames = self.num_frames
        intervals_per_chunk = math.ceil((T - 1) / num_rollouts)
        all_indices = []
        for chunk_i in range(num_rollouts):
            chunk_start = chunk_i * intervals_per_chunk
            chunk_end = min(chunk_start + intervals_per_chunk, T - 1)
            segment_len = chunk_end - chunk_start + 1
            local_indices = _uniform_resample_indices(segment_len, chunk_output_frames)
            global_indices = [chunk_start + li for li in local_indices]
            if chunk_i == 0:
                all_indices.extend(global_indices)
            else:
                all_indices.extend(global_indices[1:])
        return all_indices

    def __getitem__(self, idx):
        sample = self.samples[idx]

        first_frame = Image.open(sample["first_frame_path"]).convert("RGB")
        w, h = self.size
        first_frame = first_frame.resize((w, h), Image.BICUBIC)

        prompt = ""
        if os.path.exists(sample["instruction_path"]):
            with open(sample["instruction_path"], "r") as f:
                data = json.load(f)
            prompt = data.get("instruction", "")

        robot_only_all = self._get_robot_only_frames_all(sample)
        T = len(robot_only_all)

        chunk_interval = (self.num_frames - 1) * self.max_stride
        num_rollouts = min(
            self.max_rollouts, max(1, math.ceil((T - 1) / chunk_interval))
        )

        flat_indices = self._build_flat_indices(T, num_rollouts)
        ro_subsampled = [robot_only_all[i] for i in flat_indices]

        if self.flow_resolution is not None:
            fw, fh = self.flow_resolution
            ro_for_flow = [
                cv2.resize(f, (fw, fh), interpolation=cv2.INTER_LINEAR)
                for f in ro_subsampled
            ]
        else:
            ro_for_flow = ro_subsampled

        textured_frames = add_bg_texture(ro_for_flow)
        flows = self._compute_flows(textured_frames)
        flows = mask_flows_by_robot(flows, ro_for_flow)
        flows_resized = [self._resize_flow(f) for f in flows]
        flow_pil_list, _ = self._encode_flows_to_pil(flows_resized)

        zero_flow_pil = Image.fromarray(np.full((h, w, 3), 255, dtype=np.uint8))
        flow_pil_list.insert(0, zero_flow_pil)

        return {
            "flow_video": flow_pil_list,
            "reference_image": first_frame,
            "prompt": prompt,
            "episode_name": sample["episode_name"],
            "total_action_frames": T,
            "num_rollouts": num_rollouts,
        }


class RoboTwinCrossActionDataset(RoboTwinRolloutInferenceDataset):
    """Shuffles action HDF5 across episodes for action-following evaluation."""

    def __init__(self, *args, action_shuffle_seed=42, **kwargs):
        self._action_shuffle_seed = action_shuffle_seed
        super().__init__(*args, **kwargs)

    def _discover_episodes(self):
        samples = super()._discover_episodes()
        rng = random.Random(self._action_shuffle_seed)
        n = len(samples)
        action_paths = [s["action_hdf5"] for s in samples]
        ro_paths = [s["robot_only_hdf5"] for s in samples]
        perm = list(range(n))
        for _ in range(1000):
            rng.shuffle(perm)
            if all(perm[i] != i for i in range(n)):
                break
        for i in range(n):
            samples[i]["action_hdf5"] = action_paths[perm[i]]
            samples[i]["robot_only_hdf5"] = ro_paths[perm[i]]
        print(
            f"[CrossAction] Shuffled actions, seed={self._action_shuffle_seed}, "
            f"{sum(perm[i] != i for i in range(n))}/{n} changed"
        )
        return samples


class RoboTwinTripletActionDataset(RoboTwinRolloutInferenceDataset):
    """Uses pre-computed triplet mapping for cross-episode action assignment."""

    def __init__(self, *args, triplet_json: str = "", triplet_variant: int = 1, **kwargs):
        self._triplet_json = triplet_json
        self._triplet_variant = triplet_variant
        super().__init__(*args, **kwargs)

    def _discover_episodes(self):
        samples = super()._discover_episodes()
        with open(self._triplet_json, "r") as f:
            triplet_map = json.load(f)

        ep_name_to_idx = {s["episode_name"]: i for i, s in enumerate(samples)}
        action_paths = [s["action_hdf5"] for s in samples]
        ro_paths = [s["robot_only_hdf5"] for s in samples]

        var_key = f"variant{self._triplet_variant}_episode"
        changed = 0
        for i, sample in enumerate(samples):
            ep = sample["episode_name"]
            if ep not in triplet_map:
                continue
            entry = triplet_map[ep]
            if isinstance(entry, list):
                target_ep = entry[self._triplet_variant - 1]
            else:
                target_ep = entry[var_key]
            if target_ep not in ep_name_to_idx:
                continue
            j = ep_name_to_idx[target_ep]
            sample["action_hdf5"] = action_paths[j]
            sample["robot_only_hdf5"] = ro_paths[j]
            changed += 1

        print(
            f"[TripletAction] variant={self._triplet_variant}, "
            f"{changed}/{len(samples)} episodes remapped"
        )
        return samples


# ---------------------------------------------------------------------------
#  Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_single_chunk(
    pipe: WanVideoPipeline,
    flow_stream: nn.Module,
    context: torch.Tensor,
    ref_image: Image.Image,
    flow_frames: List[Image.Image],
    vae_z_dim: int,
    num_inference_steps: int,
    sigma_shift: float,
    seed: int,
    tiled: bool,
    progress_desc: str = "DiT denoise",
) -> List[Image.Image]:
    """Generate one chunk using the dual-stream model."""
    width, height = ref_image.size
    num_frames = len(flow_frames)

    pipe.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)

    pipe.load_models_to_device(["vae"])

    ref_tensor = pipe.preprocess_image(ref_image).transpose(0, 1)
    first_frame_z = pipe.vae.encode(
        [ref_tensor], device=pipe.device
    ).to(dtype=pipe.torch_dtype, device=pipe.device)

    flow_tensors = [pipe.preprocess_image(f).squeeze(0) for f in flow_frames]
    flow_video_tensor = torch.stack(flow_tensors, dim=1)
    flow_z = pipe.vae.encode(
        [flow_video_tensor], device=pipe.device
    ).to(dtype=pipe.torch_dtype, device=pipe.device)

    flow_first_tensor = pipe.preprocess_image(flow_frames[0]).transpose(0, 1)
    flow_fz = pipe.vae.encode(
        [flow_first_tensor], device=pipe.device
    ).to(dtype=pipe.torch_dtype, device=pipe.device)
    flow_z[:, :, 0:1] = flow_fz

    latent_length = (num_frames - 1) // 4 + 1
    h = height // pipe.vae.upsampling_factor
    w = width // pipe.vae.upsampling_factor
    shape_z = (1, vae_z_dim, latent_length, h, w)

    rgb_latents = pipe.generate_noise(shape_z, seed=seed, rand_device="cpu")
    rgb_latents = rgb_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
    rgb_latents[:, :, 0:1] = first_frame_z

    pipe.load_models_to_device(pipe.in_iteration_models)

    timesteps_iter = tqdm(
        pipe.scheduler.timesteps,
        desc=progress_desc,
        leave=False,
        dynamic_ncols=True,
        disable=False,
    )
    for progress_id, timestep in enumerate(timesteps_iter):
        timestep_tensor = timestep.unsqueeze(0).to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        rgb_pred, _flow_pred = model_fn_wan_video_dual_stream(
            dit=pipe.dit,
            flow_stream=flow_stream,
            latents=rgb_latents,
            flow_latents=flow_z,
            timestep=timestep_tensor,
            context=context,
            fuse_vae_embedding_in_latents=True,
            use_gradient_checkpointing=False,
        )
        rgb_latents = pipe.scheduler.step(
            rgb_pred, pipe.scheduler.timesteps[progress_id], rgb_latents
        )
        rgb_latents[:, :, 0:1] = first_frame_z

    pipe.load_models_to_device(["vae"])
    rgb_video = pipe.vae.decode(rgb_latents, device=pipe.device, tiled=tiled)
    rgb_frames = pipe.vae_output_to_video(rgb_video)

    pipe.load_models_to_device([])
    return rgb_frames


@torch.no_grad()
def rollout_generate(
    pipe: WanVideoPipeline,
    flow_stream: nn.Module,
    prompt: str,
    initial_frame: Image.Image,
    all_flow_frames: List[Image.Image],
    num_rollouts: int,
    chunk_size: int = 121,
    num_inference_steps: int = 50,
    sigma_shift: float = 5.0,
    seed: int = 1,
    tiled: bool = True,
) -> List[Image.Image]:
    """Generate full video with autoregressive rollout over chunks."""
    vae_z_dim = getattr(pipe.vae, "z_dim", 16)
    chunk_stride = chunk_size - 1

    pipe.load_models_to_device(["text_encoder"])
    context = pipe.prompter.encode_prompt(
        prompt, positive=True, device=pipe.device
    )

    ref_image = initial_frame
    all_frames: List[Image.Image] = []

    for chunk_idx in range(num_rollouts):
        start = chunk_idx * chunk_stride
        end = start + chunk_size
        chunk_flow = all_flow_frames[start:end]

        w, h = initial_frame.size
        n_pad = chunk_size - len(chunk_flow)
        if n_pad > 0:
            white = Image.fromarray(np.full((h, w, 3), 255, dtype=np.uint8))
            chunk_flow += [white] * n_pad

        print(
            f"  Chunk {chunk_idx + 1}/{num_rollouts}: "
            f"flow [{start}:{end}], pad={n_pad}"
        )

        chunk_frames = generate_single_chunk(
            pipe, flow_stream, context, ref_image, chunk_flow,
            vae_z_dim, num_inference_steps, sigma_shift,
            seed + chunk_idx, tiled,
            progress_desc=f"DiT denoise {chunk_idx + 1}/{num_rollouts}",
        )

        if chunk_idx == 0:
            all_frames.extend(chunk_frames)
        else:
            all_frames.extend(chunk_frames[1:])

        ref_image = chunk_frames[-1]

    return all_frames


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="FlowWAM world model inference for WorldArena evaluation."
    )
    parser.add_argument("--test_dataset_dir", type=str, required=True)
    parser.add_argument("--robot_only_dir", type=str, default=None)
    parser.add_argument("--embodiment_dir", type=str, default=None)
    parser.add_argument("--variant", type=str, default="aloha-agilex_clean_50")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="FlowWAM")
    parser.add_argument("--full_path", type=str, required=True)
    parser.add_argument("--local_model_path", type=str, default=None)
    parser.add_argument("--num_output_frames", type=int, default=121)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--size", type=int, nargs=2, default=[640, 480],
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--camera", type=str, default="head_camera")
    parser.add_argument("--flow_method", type=str, default="raft")
    parser.add_argument("--flow_device", type=str, default="cuda")
    parser.add_argument("--flow_max_magnitude", type=float, default=25.0)
    parser.add_argument("--flow_resolution", type=int, nargs=2, default=None,
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--robot_render_resolution", type=int, nargs=2,
                        default=None, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--max_stride", type=int, default=3)
    parser.add_argument("--max_rollouts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--instruction_variant", type=int, default=0,
                        choices=[0, 1, 2])
    parser.add_argument("--cross_episode_action", action="store_true")
    parser.add_argument("--action_shuffle_seed", type=int, default=42)
    parser.add_argument("--triplet_json", type=str, default="")
    parser.add_argument("--triplet_variant", type=int, default=1, choices=[1, 2])
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--episodes", type=str, nargs="+", default=None)
    # Stage-2 refiner (post-processing) options
    parser.add_argument("--refiner_ckpt_dir", type=str, default=None,
                        help="Directory containing seedvr2_ema_3b.pth and "
                             "ema_vae.pth. Defaults to refiner/SeedVR/ckpts/.")
    parser.add_argument("--refiner_alpha", type=float, default=0.7,
                        help="Blend weight: alpha * refined + (1-alpha) * stage1.")
    parser.add_argument("--refiner_res_h", type=int, default=720)
    parser.add_argument("--refiner_res_w", type=int, default=1280)
    parser.add_argument("--refiner_seed", type=int, default=666)
    parser.add_argument("--refiner_sample_steps", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    nccl_timeout = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=3600))
    accelerator = Accelerator(kwargs_handlers=[nccl_timeout])
    device = accelerator.device

    dir_suffix = (
        f"{args.model_name}_test"
        if args.instruction_variant == 0
        else f"{args.model_name}_test_{args.instruction_variant}"
    )
    video_out_dir = os.path.join(args.output_dir, dir_suffix)
    if accelerator.is_main_process:
        os.makedirs(video_out_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    size = tuple(args.size)
    flow_resolution = tuple(args.flow_resolution) if args.flow_resolution else None
    robot_render_resolution = (
        tuple(args.robot_render_resolution)
        if args.robot_render_resolution else None
    )

    dataset_kwargs = dict(
        test_dataset_dir=args.test_dataset_dir,
        robot_only_dir=args.robot_only_dir,
        camera=args.camera,
        size=size,
        num_frames=args.num_output_frames,
        flow_method=args.flow_method,
        flow_device=args.flow_device,
        flow_max_magnitude=args.flow_max_magnitude,
        embodiment_dir=args.embodiment_dir,
        variant=args.variant,
        instruction_variant=args.instruction_variant,
        flow_resolution=flow_resolution,
        robot_render_resolution=robot_render_resolution,
        max_stride=args.max_stride,
        max_rollouts=args.max_rollouts,
    )

    if args.triplet_json and os.path.exists(args.triplet_json):
        dataset = RoboTwinTripletActionDataset(
            **dataset_kwargs,
            triplet_json=args.triplet_json,
            triplet_variant=args.triplet_variant,
        )
    elif args.cross_episode_action:
        dataset = RoboTwinCrossActionDataset(
            **dataset_kwargs, action_shuffle_seed=args.action_shuffle_seed
        )
    else:
        dataset = RoboTwinRolloutInferenceDataset(**dataset_kwargs)

    if args.episodes is not None:
        ep_set = set(args.episodes)
        dataset.samples = [
            s for s in dataset.samples if s["episode_name"] in ep_set
        ]
        if accelerator.is_main_process:
            print(f"[Filter] Running only {len(dataset.samples)} episode(s)")
    elif args.max_episodes is not None and args.max_episodes < len(dataset.samples):
        dataset.samples = dataset.samples[:args.max_episodes]
        if accelerator.is_main_process:
            print(f"[QuickTest] Limiting to {args.max_episodes} episodes")

    sampler = DistributedSampler(
        dataset, num_replicas=accelerator.num_processes,
        rank=accelerator.process_index, shuffle=False, drop_last=False,
    )
    dataloader = DataLoader(
        dataset, batch_size=1, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=lambda x: x[0],
    )

    pipe, flow_stream = build_pipeline(
        device, args.full_path, args.local_model_path
    )

    from refiner.runtime import load_runner as _load_refiner
    refiner_ckpt_dir = args.refiner_ckpt_dir
    if refiner_ckpt_dir is None:
        refiner_ckpt_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models", "stage_2",
        )
    if accelerator.is_main_process:
        print(
            f"[Refiner] Loading refiner once from {refiner_ckpt_dir} "
            f"(alpha={args.refiner_alpha}, steps={args.refiner_sample_steps})",
            flush=True,
        )
    refiner = _load_refiner(
        ckpt_dir=refiner_ckpt_dir,
        sp_size=1,
        sample_steps=args.refiner_sample_steps,
        seed=args.refiner_seed,
    )
    from refiner.runtime import refine_clip as _refine_clip
    from refiner.temporal_blend import blend_arrays as _blend_arrays

    for batch in dataloader:
        episode_name = batch["episode_name"]
        prompt = batch["prompt"]
        all_flow_frames = batch["flow_video"]
        reference_image = batch["reference_image"]
        total_action_frames = batch["total_action_frames"]
        num_rollouts = batch["num_rollouts"]

        # Use plain print (not accelerator.print): Accelerate only prints on
        # local main process, so multi-GPU runs would save videos on other ranks
        # without any "Saved video" line.
        print(
            f"[Rank {accelerator.process_index}] {episode_name}: "
            f"T={total_action_frames}, num_rollouts={num_rollouts}, "
            f"{len(all_flow_frames)} flow frames",
            flush=True,
        )

        generated_frames = rollout_generate(
            pipe=pipe,
            flow_stream=flow_stream,
            prompt=prompt,
            initial_frame=reference_image,
            all_flow_frames=all_flow_frames,
            num_rollouts=num_rollouts,
            chunk_size=args.num_output_frames,
            num_inference_steps=args.num_inference_steps,
            sigma_shift=args.sigma_shift,
            seed=args.seed,
            tiled=True,
        )

        def _to_np(frames):
            return [
                np.array(f, dtype=np.uint8) if isinstance(f, Image.Image)
                else np.clip(f, 0, 255).astype(np.uint8)
                for f in frames
            ]

        if len(generated_frames) > args.num_output_frames:
            indices = _uniform_resample_indices(
                len(generated_frames), args.num_output_frames
            )
            output_frames = [generated_frames[i] for i in indices]
        else:
            output_frames = generated_frames

        out_np = np.stack(_to_np(output_frames), axis=0)

        try:
            refined_np = _refine_clip(
                refiner,
                out_np,
                res_h=args.refiner_res_h,
                res_w=args.refiner_res_w,
            )
            out_np = _blend_arrays(
                refined_np, out_np, alpha=args.refiner_alpha
            )
        except Exception as e:
            print(
                f"[Rank {accelerator.process_index}] {episode_name}: "
                f"refiner failed ({type(e).__name__}: {e}); "
                f"falling back to Stage-1 output",
                flush=True,
            )

        out_path = os.path.join(video_out_dir, f"{episode_name}.mp4")
        imageio.mimwrite(out_path, list(out_np), fps=args.fps)
        print(
            f"[Rank {accelerator.process_index}] Saved video: {out_path}",
            flush=True,
        )

        print(
            f"[Rank {accelerator.process_index}] {episode_name}: "
            f"{len(generated_frames)}f generated -> "
            f"{len(output_frames)}f output -> {out_path}",
            flush=True,
        )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        n_sub = len([f for f in os.listdir(video_out_dir) if f.endswith(".mp4")])
        print(f"[Done] {n_sub} videos -> {video_out_dir}")


if __name__ == "__main__":
    main()
