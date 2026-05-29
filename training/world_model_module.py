"""
Dual-stream world model training module for RoboTwin.

Two latent streams share one Wan DiT backbone:
  - RGB stream:  head-camera frames        -> dit.patch_embedding
  - Flow stream: robot-only optical flow    -> flow_stream.flow_patch_embedding
joined by self-attention and supervised with a weighted MSE on both streams.

Training is autoregressive: each episode is split into one or more overlapping
chunks of ``chunk_output_frames``. Every chunk is denoised using the previous
chunk's decoded last frame as its conditioning reference, so the model learns
to roll its own predictions forward.
"""

import os, re, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule
from diffsynth.models.wan_video_dit_dual_stream import init_flow_stream
from diffsynth.pipelines.wan_video_dual_stream import model_fn_wan_video_dual_stream

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanDualStreamWorldModelModule(DiffusionTrainingModule):
    """Dual-stream (RGB + flow) world model with autoregressive rollout.

    Handles checkpoint / LoRA loading, FlowStream init, and optional fp32
    modulation upcasting, then trains with a per-chunk AR rollout where each
    chunk's reference frame is the decoded last frame of the previous chunk.
    """

    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None, audio_processor_config=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        resume_checkpoint=None,
        flow_loss_weight=0.5,
        ref_aug_strength=0.1,
        fp32_modulation=True,
        chunk_output_frames=121,
    ):
        super().__init__()
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        if audio_processor_config is not None:
            audio_processor_config = ModelConfig(
                model_id=audio_processor_config.split(":")[0],
                origin_file_pattern=audio_processor_config.split(":")[1],
            )
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device="cpu",
            model_configs=model_configs,
            audio_processor_config=audio_processor_config,
        )

        self.vae_z_dim = getattr(self.pipe.vae, "z_dim", 16)
        self.lora_mode = lora_base_model is not None

        self.flow_stream = init_flow_stream(self.pipe.dit)

        lora_resume_state = None
        if resume_checkpoint is not None:
            if self.lora_mode:
                lora_resume_state = self._load_lora_resume_checkpoint_pre(resume_checkpoint)
            else:
                self._load_resume_checkpoint(resume_checkpoint)

        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        if self.lora_mode:
            self._unfreeze_dual_stream_modules()
            if lora_resume_state is not None:
                self._load_lora_resume_checkpoint_post(lora_resume_state)

        if fp32_modulation:
            self._upcast_modulation_to_fp32()

        self.ref_aug_strength = ref_aug_strength
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.flow_loss_weight = flow_loss_weight
        self.chunk_output_frames = chunk_output_frames

    def _load_resume_checkpoint(self, ckpt_path):
        print(f"[Resume] Loading checkpoint: {ckpt_path}")
        state_dict = load_state_dict(ckpt_path)
        dit_keys = {}
        flow_keys = {}
        for k, v in state_dict.items():
            if k.startswith("flow_stream."):
                flow_keys[k.replace("flow_stream.", "")] = v
            else:
                dit_keys[k] = v
        if dit_keys:
            self.pipe.dit.load_state_dict(dit_keys, strict=False)
            print(f"[Resume] Loaded {len(dit_keys)} DiT keys")
        if flow_keys:
            self.flow_stream.load_state_dict(flow_keys, strict=False)
            print(f"[Resume] Loaded {len(flow_keys)} FlowStream keys")

    def _unfreeze_dual_stream_modules(self):
        self.flow_stream.requires_grad_(True)
        self.flow_stream.train()
        n_flow = sum(p.numel() for p in self.flow_stream.parameters())

        modulation_nn_modules = [
            self.pipe.dit.time_projection,
            self.pipe.dit.time_embedding,
        ]
        for module in modulation_nn_modules:
            module.requires_grad_(True)
            module.train()

        modulation_params = []
        for block in self.pipe.dit.blocks:
            if hasattr(block, 'modulation') and isinstance(block.modulation, nn.Parameter):
                block.modulation.requires_grad = True
                modulation_params.append(block.modulation)
        if hasattr(self.pipe.dit.head, 'modulation') and isinstance(self.pipe.dit.head.modulation, nn.Parameter):
            self.pipe.dit.head.modulation.requires_grad = True
            modulation_params.append(self.pipe.dit.head.modulation)

        n_mod_nn = sum(p.numel() for m in modulation_nn_modules for p in m.parameters())
        n_mod_param = sum(p.numel() for p in modulation_params)
        print(f"[LoRA+DualStream] Unfroze flow_stream ({n_flow:,} params)")
        print(f"[LoRA+Modulation] Unfroze modulation ({n_mod_nn + n_mod_param:,} params)")

    @staticmethod
    def _register_fp32_hooks(module):
        module.float()

        def pre_hook(_mod, args):
            return tuple(a.float() if isinstance(a, torch.Tensor) else a for a in args)

        def post_hook(_mod, _args, output):
            return output.bfloat16() if isinstance(output, torch.Tensor) else output

        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(post_hook)

    def _upcast_modulation_to_fp32(self):
        upcasted = 0
        for block in self.pipe.dit.blocks:
            if hasattr(block, 'modulation') and isinstance(block.modulation, nn.Parameter):
                block.modulation.data = block.modulation.data.float()
                upcasted += block.modulation.numel()
        if hasattr(self.pipe.dit.head, 'modulation') and isinstance(self.pipe.dit.head.modulation, nn.Parameter):
            self.pipe.dit.head.modulation.data = self.pipe.dit.head.modulation.data.float()
            upcasted += self.pipe.dit.head.modulation.numel()

        n_hooked = 0
        for module in [self.pipe.dit.time_embedding, self.pipe.dit.time_projection]:
            self._register_fp32_hooks(module)
            upcasted += sum(p.numel() for p in module.parameters())
            n_hooked += 1

        for module in self.pipe.dit.modules():
            if isinstance(module, nn.LayerNorm):
                self._register_fp32_hooks(module)
                upcasted += sum(p.numel() for p in module.parameters())
                n_hooked += 1

        print(f"[FP32Modulation] Upcasted {upcasted:,} params to float32 "
              f"({n_hooked} modules with FP32 hooks)")

    def _load_lora_resume_checkpoint_pre(self, ckpt_path):
        print(f"[LoRA Resume] Loading checkpoint: {ckpt_path}")
        state_dict = load_state_dict(ckpt_path)
        lora_keys = {k: v for k, v in state_dict.items()
                     if "lora_A" in k or "lora_B" in k}
        module_keys = {k: v for k, v in state_dict.items()
                       if k not in lora_keys}
        dit_keys = {}
        flow_keys = {}
        for k, v in module_keys.items():
            if k.startswith("flow_stream."):
                flow_keys[k.replace("flow_stream.", "")] = v
            else:
                dit_keys[k] = v
        if dit_keys:
            self.pipe.dit.load_state_dict(dit_keys, strict=False)
        if flow_keys:
            self.flow_stream.load_state_dict(flow_keys, strict=False)
        print(f"[LoRA Resume] Deferred {len(lora_keys)} LoRA keys")
        return lora_keys

    def _load_lora_resume_checkpoint_post(self, lora_state_dict):
        mapped = self.mapping_lora_state_dict(lora_state_dict)
        self.pipe.dit.load_state_dict(mapped, strict=False)
        print(f"[LoRA Resume Post] Loaded {len(mapped)} LoRA keys into DiT")

    def _encode_video_to_latents(self, pil_frames):
        tensors = []
        for f in pil_frames:
            t = self.pipe.preprocess_image(f).squeeze(0)
            tensors.append(t)
        video_tensor = torch.stack(tensors, dim=1)
        return self.pipe.vae.encode(
            [video_tensor], device=self.pipe.device, progress_bar=False
        ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

    def _encode_single_frame(self, pil_img, w, h):
        img_tensor = self.pipe.preprocess_image(
            pil_img.resize((w, h))
        ).transpose(0, 1)
        return self.pipe.vae.encode(
            [img_tensor], device=self.pipe.device, progress_bar=False
        ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

    def forward(self, data):
        num_rollouts = data["num_rollouts"]
        chunk_stride = self.chunk_output_frames - 1

        self.pipe.load_models_to_device(["text_encoder"])
        context = self.pipe.prompter.encode_prompt(
            data["prompt"], positive=True, device=self.pipe.device
        )

        current_ref_pil = data["reference_image"]
        loss_dict = {}
        loss_weight = 1.0 / num_rollouts

        for i in range(num_rollouts):
            start = i * chunk_stride
            end = start + self.chunk_output_frames

            chunk_rgb_pils = data["rgb_video"][start:end]
            chunk_flow_pils = data["flow_video"][start:end]

            h_px, w_px = chunk_rgb_pils[0].size[1], chunk_rgb_pils[0].size[0]

            self.pipe.load_models_to_device(["vae"])
            rgb_z = self._encode_video_to_latents(chunk_rgb_pils)
            flow_z = self._encode_video_to_latents(chunk_flow_pils)
            rgb_fz = self._encode_single_frame(current_ref_pil, w_px, h_px)
            flow_fz = self._encode_single_frame(chunk_flow_pils[0], w_px, h_px)

            rgb_z[:, :, 0:1] = rgb_fz
            flow_z[:, :, 0:1] = flow_fz

            rgb_noise = torch.randn_like(rgb_z)
            rgb_noise[:, :, 0:1] = rgb_fz

            flow_noise = torch.zeros_like(flow_z)

            max_tb = int(self.max_timestep_boundary * self.pipe.scheduler.num_train_timesteps)
            min_tb = int(self.min_timestep_boundary * self.pipe.scheduler.num_train_timesteps)
            timestep_id = torch.randint(min_tb, max_tb, (1,))
            timestep = self.pipe.scheduler.timesteps[timestep_id].to(
                dtype=self.pipe.torch_dtype, device=self.pipe.device
            )

            if self.ref_aug_strength > 0:
                aug_scale = random.random() * self.ref_aug_strength
                aug_ffl = rgb_fz + aug_scale * torch.randn_like(rgb_fz)
                rgb_noise[:, :, 0:1] = aug_ffl
                rgb_z[:, :, 0:1] = aug_ffl

            rgb_noisy = self.pipe.scheduler.add_noise(rgb_z, rgb_noise, timestep)

            rgb_target = self.pipe.scheduler.training_target(rgb_z, rgb_noise, timestep)
            flow_target = self.pipe.scheduler.training_target(flow_z, flow_noise, timestep)

            self.pipe.load_models_to_device(self.pipe.in_iteration_models)
            rgb_pred, flow_pred = model_fn_wan_video_dual_stream(
                dit=self.pipe.dit,
                flow_stream=self.flow_stream,
                latents=rgb_noisy,
                flow_latents=flow_z,
                timestep=timestep,
                context=context,
                fuse_vae_embedding_in_latents=True,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            )

            loss_rgb = F.mse_loss(
                rgb_pred[:, :, 1:].float(), rgb_target[:, :, 1:].float()
            )
            loss_flow = F.mse_loss(
                flow_pred.float(), flow_target.float()
            )

            w = self.flow_loss_weight
            if w > 0:
                chunk_loss = (1.0 - w) * loss_rgb + w * loss_flow
            else:
                chunk_loss = loss_rgb

            loss_dict[f"loss_roll_{i}"] = chunk_loss * loss_weight
            loss_dict[f"loss_rgb_{i}"] = loss_rgb.detach()
            loss_dict[f"loss_flow_{i}"] = loss_flow.detach()

            # Hand the decoded last frame to the next chunk as its reference.
            if i < num_rollouts - 1:
                with torch.no_grad():
                    sigma = timestep / self.pipe.scheduler.num_train_timesteps
                    pred_x0_rgb = rgb_noisy - rgb_pred * sigma.view(-1, 1, 1, 1, 1)

                    self.pipe.load_models_to_device(["vae"])
                    decoded_frames = self.pipe.vae_output_to_video(
                        self.pipe.vae.decode(
                            pred_x0_rgb, device=self.pipe.device,
                            tiled=True, progress_bar=False,
                        )
                    )
                    current_ref_pil = decoded_frames[-1]

        total_loss = sum(v for k, v in loss_dict.items() if k.startswith("loss_roll_"))
        loss_dict["loss"] = total_loss
        loss_dict["loss_rgb"] = sum(
            loss_dict[f"loss_rgb_{i}"] for i in range(num_rollouts)
        ) / num_rollouts
        loss_dict["loss_flow"] = sum(
            loss_dict[f"loss_flow_{i}"] for i in range(num_rollouts)
        ) / num_rollouts

        return loss_dict


def parse_start_epoch(resume_checkpoint):
    if resume_checkpoint is None:
        return 0
    basename = os.path.basename(resume_checkpoint)
    m = re.match(r"epoch-(\d+)", basename)
    if m:
        return int(m.group(1)) + 1
    return 0
