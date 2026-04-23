"""Stage-2 refiner runtime: thin wrapper around the vendored SeedVR2 inference.

Exposes two APIs to the Stage-1 main process:

    runner = load_runner(ckpt_dir, sp_size=1)
    refined_uint8 = refine_clip(runner, frames_uint8, fps=24, seed=666)

`refine_clip` accepts and returns a (T, H, W, 3) uint8 ndarray (RGB) and
performs no disk I/O. Models are loaded once and cached on the runner.

This module isolates SeedVR's relative-path quirks (it `cd`'s into the
vendored `SeedVR/` directory while loading models / configs) and restores
the caller's working directory afterwards.
"""
from __future__ import annotations

import gc
import os
import sys
import datetime
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
from einops import rearrange


_REFINER_DIR = os.path.dirname(os.path.abspath(__file__))
_SEEDVR_DIR = os.path.join(_REFINER_DIR, "SeedVR")
_DEFAULT_CKPT_DIR = os.path.normpath(
    os.path.join(_REFINER_DIR, "..", "models", "stage_2")
)
_DIT_CKPT_NAME = "seedvr2_ema_3b.pth"
_VAE_CKPT_NAME = "ema_vae.pth"


@contextmanager
def _chdir_seedvr():
    """Temporarily chdir into vendored SeedVR/ so its `./configs_3b`,
    `./ckpts/`, `pos_emb.pt`, `neg_emb.pt` relative paths resolve."""
    old_cwd = os.getcwd()
    old_path = list(sys.path)
    try:
        os.chdir(_SEEDVR_DIR)
        if _SEEDVR_DIR not in sys.path:
            sys.path.insert(0, _SEEDVR_DIR)
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def _ensure_distributed_initialized():
    """Ensure dist is initialised without clobbering the caller's CUDA settings.

    SeedVR's own ``init_torch`` changes global state (TF32, cuDNN benchmark).
    We only touch the process group here and leave backend flags alone so the
    Stage-1 pipeline's behaviour is not affected.
    """
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    if not dist.is_available() or not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group(
            backend="nccl",
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
            timeout=datetime.timedelta(seconds=3600),
        )


class RefinerRunner:
    """Holds SeedVR's `VideoDiffusionInfer` plus cached text embeddings."""

    def __init__(
        self,
        runner,
        text_pos_embeds: torch.Tensor,
        text_neg_embeds: torch.Tensor,
        sp_size: int,
        sample_steps: int,
        cfg_scale: float,
        cfg_rescale: float,
        seed: int,
    ):
        self.runner = runner
        self.text_pos_embeds = text_pos_embeds
        self.text_neg_embeds = text_neg_embeds
        self.sp_size = sp_size
        self.sample_steps = sample_steps
        self.cfg_scale = cfg_scale
        self.cfg_rescale = cfg_rescale
        self.seed = seed

    def offload_to_cpu(self):
        """Move refiner DiT + VAE to CPU to free GPU memory."""
        self.runner.dit.cpu()
        self.runner.vae.cpu()
        self.text_pos_embeds = self.text_pos_embeds.cpu()
        self.text_neg_embeds = self.text_neg_embeds.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    def onload_to_gpu(self):
        """Move refiner DiT + VAE back to the CUDA device."""
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        self.runner.dit.to(device)
        self.runner.vae.to(device=device, dtype=torch.bfloat16)
        self.text_pos_embeds = self.text_pos_embeds.to(device)
        self.text_neg_embeds = self.text_neg_embeds.to(device)


def load_runner(
    ckpt_dir: Optional[str] = None,
    sp_size: int = 1,
    sample_steps: int = 1,
    cfg_scale: float = 1.0,
    cfg_rescale: float = 0.0,
    seed: int = 666,
) -> RefinerRunner:
    """Load Stage-2 refiner once and return a reusable runner.

    `ckpt_dir` defaults to `inference/models/stage_2/`, which must contain
    `flowwam_stage2_dit.pth` and `flowwam_stage2_vae.pth`.
    """
    _ensure_distributed_initialized()

    if ckpt_dir is None:
        ckpt_dir = _DEFAULT_CKPT_DIR
    dit_ckpt = os.path.abspath(os.path.join(ckpt_dir, _DIT_CKPT_NAME))
    vae_ckpt = os.path.abspath(os.path.join(ckpt_dir, _VAE_CKPT_NAME))
    if not os.path.isfile(dit_ckpt):
        raise FileNotFoundError(f"Stage-2 DiT ckpt not found: {dit_ckpt}")
    if not os.path.isfile(vae_ckpt):
        raise FileNotFoundError(f"Stage-2 VAE ckpt not found: {vae_ckpt}")

    with _chdir_seedvr():
        from omegaconf import OmegaConf
        from common.config import load_config
        from common.distributed.advanced import init_sequence_parallel
        from projects.video_diffusion_sr.infer import VideoDiffusionInfer

        if sp_size > 1:
            init_sequence_parallel(sp_size)

        config = load_config(os.path.join("./configs_3b", "main.yaml"))
        runner = VideoDiffusionInfer(config)
        OmegaConf.set_readonly(runner.config, False)

        runner.config.vae.checkpoint = vae_ckpt

        runner.configure_dit_model(device="cuda", checkpoint=dit_ckpt)
        runner.configure_vae_model()
        if hasattr(runner.vae, "set_memory_limit"):
            runner.vae.set_memory_limit(**runner.config.vae.memory_limit)

        runner.config.diffusion.cfg.scale = cfg_scale
        runner.config.diffusion.cfg.rescale = cfg_rescale
        runner.config.diffusion.timesteps.sampling.steps = sample_steps
        runner.configure_diffusion()

        text_pos_embeds = torch.load("pos_emb.pt")
        text_neg_embeds = torch.load("neg_emb.pt")

    gc.collect()
    torch.cuda.empty_cache()

    ref = RefinerRunner(
        runner=runner,
        text_pos_embeds=text_pos_embeds,
        text_neg_embeds=text_neg_embeds,
        sp_size=sp_size,
        sample_steps=sample_steps,
        cfg_scale=cfg_scale,
        cfg_rescale=cfg_rescale,
        seed=seed,
    )
    return ref


def _cut_videos(videos: torch.Tensor, sp_size: int) -> torch.Tensor:
    """Pad time dim so (T-1) % (4*sp_size) == 0. Mirrors SeedVR upstream."""
    t = videos.size(1)
    if t == 1:
        return videos
    if t <= 4 * sp_size:
        padding = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - t + 1)
        return torch.cat([videos] + padding, dim=1)
    if (t - 1) % (4 * sp_size) == 0:
        return videos
    padding = [videos[:, -1].unsqueeze(1)] * (
        4 * sp_size - ((t - 1) % (4 * sp_size))
    )
    return torch.cat([videos] + padding, dim=1)


@torch.no_grad()
def refine_clip(
    runner: RefinerRunner,
    frames: np.ndarray,
    res_h: int = 720,
    res_w: int = 1280,
) -> np.ndarray:
    """Run Stage-2 SeedVR2 refinement on a single clip.

    Args:
        frames: (T, H, W, 3) uint8 ndarray, RGB.
        res_h, res_w: target effective resolution; SeedVR re-scales internally
            to area=sqrt(res_h*res_w), preserving aspect.

    Returns:
        (T, H', W', 3) uint8 ndarray, RGB. T is preserved; H'/W' are
        determined by SeedVR's NaResize and DivisibleCrop transforms.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expect (T,H,W,3) uint8, got {frames.shape}")

    return _refine_clip_inner(runner, frames, res_h, res_w)


@torch.no_grad()
def _refine_clip_inner(
    runner: RefinerRunner,
    frames: np.ndarray,
    res_h: int,
    res_w: int,
) -> np.ndarray:
    with _chdir_seedvr():
        from torchvision.transforms import Compose, Lambda, Normalize
        from data.image.transforms.divisible_crop import DivisibleCrop
        from data.image.transforms.na_resize import NaResize
        from data.video.transforms.rearrange import Rearrange
        from common.distributed import get_device
        from common.distributed.ops import sync_data
        from common.seed import set_seed

        try:
            from projects.video_diffusion_sr.color_fix import (
                wavelet_reconstruction,
            )
            use_colorfix = True
        except Exception:
            use_colorfix = False

        device = get_device()
        set_seed(runner.seed, same_across_ranks=True)

        # (T,H,W,3) uint8 -> (T,3,H,W) float in [0,1]
        video = torch.from_numpy(frames).to(device)
        video = rearrange(video, "t h w c -> t c h w").float() / 255.0

        transform = Compose([
            NaResize(
                resolution=(res_h * res_w) ** 0.5,
                mode="area",
                downsample_only=False,
            ),
            Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            DivisibleCrop((16, 16)),
            Normalize(0.5, 0.5),
            Rearrange("t c h w -> c t h w"),
        ])

        cond = transform(video)
        ori_length = cond.size(1)
        input_video = cond
        cond_padded = _cut_videos(cond, runner.sp_size)

        cond_latents = runner.runner.vae_encode([cond_padded])

        text_pos = runner.text_pos_embeds.to(device)
        text_neg = runner.text_neg_embeds.to(device)
        text_embeds_dict = {
            "texts_pos": [text_pos],
            "texts_neg": [text_neg],
        }

        noises = [torch.randn_like(latent) for latent in cond_latents]
        aug_noises = [torch.randn_like(latent) for latent in cond_latents]
        noises, aug_noises, cond_latents = sync_data(
            (noises, aug_noises, cond_latents), 0
        )
        noises = [n.to(device) for n in noises]
        aug_noises = [n.to(device) for n in aug_noises]
        cond_latents = [c.to(device) for c in cond_latents]

        cond_noise_scale = 0.0

        def _add_noise(x, aug_noise):
            t = torch.tensor([1000.0], device=device) * cond_noise_scale
            shape = torch.tensor(x.shape[1:], device=device)[None]
            t = runner.runner.timestep_transform(t, shape)
            return runner.runner.schedule.forward(x, aug_noise, t)

        conditions = [
            runner.runner.get_condition(
                noise,
                task="sr",
                latent_blur=_add_noise(latent_blur, aug_noise),
            )
            for noise, aug_noise, latent_blur in zip(
                noises, aug_noises, cond_latents
            )
        ]

        with torch.autocast("cuda", torch.bfloat16, enabled=True):
            video_tensors = runner.runner.inference(
                noises=noises,
                conditions=conditions,
                dit_offload=False,
                **text_embeds_dict,
            )

        sample = video_tensors[0]
        sample = (
            rearrange(sample[:, None], "c t h w -> t c h w")
            if sample.ndim == 3
            else rearrange(sample, "c t h w -> t c h w")
        )
        if ori_length < sample.shape[0]:
            sample = sample[:ori_length]

        if use_colorfix:
            input_thwc = (
                rearrange(input_video[:, None], "c t h w -> t c h w")
                if input_video.ndim == 3
                else rearrange(input_video, "c t h w -> t c h w")
            )
            sample = wavelet_reconstruction(
                sample, input_thwc[: sample.size(0)].to(sample.device)
            )

        sample = sample.to("cpu")
        sample = (
            rearrange(sample[:, None], "t c h w -> t h w c")
            if sample.ndim == 3
            else rearrange(sample, "t c h w -> t h w c")
        )
        sample = sample.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round()
        out = sample.to(torch.uint8).numpy()

    gc.collect()
    torch.cuda.empty_cache()
    return out
