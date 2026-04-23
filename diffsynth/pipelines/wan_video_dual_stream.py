"""
Dual-stream forward function for RGB + Flow joint self-attention.

Does NOT touch wan_video_new.py. Used by the dual-stream training module
and inference server as a drop-in replacement for model_fn_wan_video.

Design:
    1. RGB latents  -> dit.patch_embedding  -> rgb_tokens
    2. Flow latents -> flow_stream.flow_patch_embedding -> flow_tokens
    3. RoPE computed independently for both RGB and Flow grids.
    4. Each DiT block: joint self-attn([rgb, flow]), separate cross-attn,
       shared FFN.
    5. dit.head(rgb_tokens) -> rgb_pred
       flow_stream.flow_head(flow_tokens) -> flow_pred
"""

import torch
from einops import rearrange
from ..models.wan_video_dit import (
    sinusoidal_embedding_1d,
    modulate,
    rope_apply,
    flash_attention,
)


def _dual_stream_block_fn(
    block, rgb_tokens, flow_tokens, context, t_mod, rgb_freqs, flow_freqs, n_rgb,
):
    """One DiT block with dual-stream joint self-attention.

    Joint self-attention concatenates [rgb, flow] along the sequence
    dimension. RoPE is applied independently to both RGB and Flow tokens
    using their respective positional frequencies.

    Cross-attention and FFN use the same block weights for both streams.
    """
    has_seq = len(t_mod.shape) == 4
    chunk_dim = 2 if has_seq else 1

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
    ).chunk(6, dim=chunk_dim)

    if has_seq:
        shift_msa, scale_msa, gate_msa = (
            shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
        )
        shift_mlp, scale_mlp, gate_mlp = (
            shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
        )
        shift_msa_r, shift_msa_f = shift_msa[:, :n_rgb], shift_msa[:, n_rgb:]
        scale_msa_r, scale_msa_f = scale_msa[:, :n_rgb], scale_msa[:, n_rgb:]
        gate_msa_r, gate_msa_f = gate_msa[:, :n_rgb], gate_msa[:, n_rgb:]
        shift_mlp_r, shift_mlp_f = shift_mlp[:, :n_rgb], shift_mlp[:, n_rgb:]
        scale_mlp_r, scale_mlp_f = scale_mlp[:, :n_rgb], scale_mlp[:, n_rgb:]
        gate_mlp_r, gate_mlp_f = gate_mlp[:, :n_rgb], gate_mlp[:, n_rgb:]
    else:
        shift_msa_r = shift_msa_f = shift_msa
        scale_msa_r = scale_msa_f = scale_msa
        gate_msa_r = gate_msa_f = gate_msa
        shift_mlp_r = shift_mlp_f = shift_mlp
        scale_mlp_r = scale_mlp_f = scale_mlp
        gate_mlp_r = gate_mlp_f = gate_mlp

    # ---- Self-Attention: joint [rgb, flow] ----
    rgb_inp = modulate(block.norm1(rgb_tokens), shift_msa_r, scale_msa_r)
    flow_inp = modulate(block.norm1(flow_tokens), shift_msa_f, scale_msa_f)

    sa = block.self_attn
    rgb_q = sa.norm_q(sa.q(rgb_inp))
    rgb_k = sa.norm_k(sa.k(rgb_inp))
    rgb_v = sa.v(rgb_inp)

    flow_q = sa.norm_q(sa.q(flow_inp))
    flow_k = sa.norm_k(sa.k(flow_inp))
    flow_v = sa.v(flow_inp)

    rgb_q = rope_apply(rgb_q, rgb_freqs, sa.num_heads)
    rgb_k = rope_apply(rgb_k, rgb_freqs, sa.num_heads)

    flow_q = rope_apply(flow_q, flow_freqs, sa.num_heads)
    flow_k = rope_apply(flow_k, flow_freqs, sa.num_heads)

    q = torch.cat([rgb_q, flow_q], dim=1)
    k = torch.cat([rgb_k, flow_k], dim=1)
    v = torch.cat([rgb_v, flow_v], dim=1)

    attn_out = sa.o(flash_attention(q, k, v, sa.num_heads))
    rgb_tokens = block.gate(rgb_tokens, gate_msa_r, attn_out[:, :n_rgb])
    flow_tokens = block.gate(flow_tokens, gate_msa_f, attn_out[:, n_rgb:])

    # ---- Cross-Attention (text context, both streams) ----
    rgb_tokens = rgb_tokens + block.cross_attn(block.norm3(rgb_tokens), context)
    flow_tokens = flow_tokens + block.cross_attn(block.norm3(flow_tokens), context)

    # ---- FFN (shared weights, both streams) ----
    rgb_ff = modulate(block.norm2(rgb_tokens), shift_mlp_r, scale_mlp_r)
    flow_ff = modulate(block.norm2(flow_tokens), shift_mlp_f, scale_mlp_f)
    rgb_tokens = block.gate(rgb_tokens, gate_mlp_r, block.ffn(rgb_ff))
    flow_tokens = block.gate(flow_tokens, gate_mlp_f, block.ffn(flow_ff))

    return rgb_tokens, flow_tokens


def model_fn_wan_video_dual_stream(
    dit,
    flow_stream,
    latents,
    flow_latents,
    timestep,
    context,
    fuse_vae_embedding_in_latents=False,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    """Dual-stream forward: RGB + Flow with joint self-attention.

    Args:
        dit: WanModel (pretrained, unmodified).
        flow_stream: FlowStreamModule with flow_patch_embedding + flow_head.
        latents: (B, C, T, H_r, W_r) RGB stream latents (noisy).
        flow_latents: (B, C, T, H_f, W_f) Flow stream latents (noisy).
        timestep: (B,) diffusion timestep.
        context: (B, S, text_dim) raw text embeddings (before text_embedding).
        fuse_vae_embedding_in_latents: if True, per-token timestep where
            first-frame tokens receive t=0 (TI2V-5B style).

    Returns:
        (rgb_pred, flow_pred): each in latent space, same shape as inputs.
    """
    B = latents.shape[0]

    # ---- Timestep ----
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        rgb_spatial = latents.shape[3] * latents.shape[4] // 4
        rgb_temporal = latents.shape[2]
        flow_spatial = flow_latents.shape[3] * flow_latents.shape[4] // 4
        flow_temporal = flow_latents.shape[2]

        t_per_token_list = []
        for b in range(B):
            ts_b = (
                timestep[b]
                if timestep.dim() >= 1 and timestep.shape[0] > 1
                else timestep
            )
            rgb_tpt = torch.cat([
                torch.zeros(1, rgb_spatial, dtype=latents.dtype,
                            device=latents.device),
                torch.ones(rgb_temporal - 1, rgb_spatial, dtype=latents.dtype,
                           device=latents.device) * ts_b,
            ]).flatten()
            flow_tpt = torch.cat([
                torch.zeros(1, flow_spatial, dtype=latents.dtype,
                            device=latents.device),
                torch.ones(flow_temporal - 1, flow_spatial, dtype=latents.dtype,
                           device=latents.device) * ts_b,
            ]).flatten()
            t_per_token_list.append(torch.cat([rgb_tpt, flow_tpt]))

        t_per_token = torch.stack(t_per_token_list, dim=0)
        t = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, t_per_token.reshape(-1))
            .reshape(B, -1, dit.freq_dim)
        )
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))

        n_rgb_tok = rgb_spatial * rgb_temporal
        t_rgb = t[:, :n_rgb_tok]
        t_flow = t[:, n_rgb_tok:]
    else:
        t = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype)
        )
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
        t_rgb = t
        t_flow = t

    # ---- Context ----
    context = dit.text_embedding(context)

    # ---- Patchify ----
    rgb_5d = dit.patchify(latents)
    f_r, h_r, w_r = rgb_5d.shape[2:]
    rgb_tokens = rearrange(rgb_5d, 'b c f h w -> b (f h w) c').contiguous()
    n_rgb = rgb_tokens.shape[1]

    flow_5d = flow_stream.patchify(flow_latents)
    f_f, h_f, w_f = flow_5d.shape[2:]
    flow_tokens = rearrange(flow_5d, 'b c f h w -> b (f h w) c').contiguous()

    # ---- RoPE (both RGB and Flow get independent positional encoding) ----
    rgb_freqs = torch.cat([
        dit.freqs[0][:f_r].view(f_r, 1, 1, -1).expand(f_r, h_r, w_r, -1),
        dit.freqs[1][:h_r].view(1, h_r, 1, -1).expand(f_r, h_r, w_r, -1),
        dit.freqs[2][:w_r].view(1, 1, w_r, -1).expand(f_r, h_r, w_r, -1),
    ], dim=-1).reshape(f_r * h_r * w_r, 1, -1).to(rgb_tokens.device)

    flow_freqs = torch.cat([
        dit.freqs[0][:f_f].view(f_f, 1, 1, -1).expand(f_f, h_f, w_f, -1),
        dit.freqs[1][:h_f].view(1, h_f, 1, -1).expand(f_f, h_f, w_f, -1),
        dit.freqs[2][:w_f].view(1, 1, w_f, -1).expand(f_f, h_f, w_f, -1),
    ], dim=-1).reshape(f_f * h_f * w_f, 1, -1).to(flow_tokens.device)

    # ---- Block loop ----
    def _make_ckpt_fn(block):
        def fn(rtok, ftok, ctx, tm, rfreqs, ffreqs):
            return _dual_stream_block_fn(
                block, rtok, ftok, ctx, tm, rfreqs, ffreqs, n_rgb,
            )
        return fn

    for block in dit.blocks:
        if use_gradient_checkpointing and dit.training:
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    rgb_tokens, flow_tokens = (
                        torch.utils.checkpoint.checkpoint(
                            _make_ckpt_fn(block),
                            rgb_tokens, flow_tokens, context,
                            t_mod, rgb_freqs, flow_freqs,
                            use_reentrant=False,
                        )
                    )
            else:
                rgb_tokens, flow_tokens = (
                    torch.utils.checkpoint.checkpoint(
                        _make_ckpt_fn(block),
                        rgb_tokens, flow_tokens, context,
                        t_mod, rgb_freqs, flow_freqs,
                        use_reentrant=False,
                    )
                )
        else:
            rgb_tokens, flow_tokens = _dual_stream_block_fn(
                block, rgb_tokens, flow_tokens,
                context, t_mod, rgb_freqs, flow_freqs, n_rgb,
            )

    # ---- Heads ----
    rgb_out = dit.head(rgb_tokens, t_rgb)
    rgb_out = dit.unpatchify(rgb_out, (f_r, h_r, w_r))

    flow_out = flow_stream.apply_head(flow_tokens, t_flow)
    flow_out = flow_stream.unpatchify(flow_out, (f_f, h_f, w_f))

    return rgb_out, flow_out
