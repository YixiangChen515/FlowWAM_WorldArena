"""
Dual-stream flow components for WanModel.

FlowStreamModule is a standalone nn.Module providing flow_patch_embedding
+ flow_head, deep-copied from the pretrained DiT. Does NOT modify WanModel.

RGB goes through dit.patch_embedding / dit.head (unchanged).
Flow goes through FlowStreamModule.flow_patch_embedding / .flow_head.
"""

import copy
import torch
import torch.nn as nn
from einops import rearrange


class FlowStreamModule(nn.Module):
    """Flow stream's patch embedding and output head.

    Architecture mirrors the pretrained DiT's patch_embedding and head.
    Weights are deep-copied so both streams start from identical init.
    """

    def __init__(self, dit):
        super().__init__()
        self.patch_size = tuple(dit.patch_size)
        self.flow_patch_embedding = copy.deepcopy(dit.patch_embedding)
        self.flow_head = copy.deepcopy(dit.head)

    def patchify(self, flow_latent):
        """Apply flow patch embedding.

        Returns:
            x: (B, dim, f, h, w) 5-D feature tensor
        """
        return self.flow_patch_embedding(flow_latent)

    def unpatchify(self, x, grid_size):
        """Reverse patchification from token to latent space."""
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2],
        )

    def apply_head(self, flow_tokens, t):
        """Apply flow head to predict flow velocity."""
        return self.flow_head(flow_tokens, t)


def init_flow_stream(dit):
    """Create FlowStreamModule initialized from pretrained DiT weights."""
    module = FlowStreamModule(dit)
    n_params = sum(p.numel() for p in module.parameters())
    pe = dit.patch_embedding
    print(
        f"[FlowStream] Created FlowStreamModule: {n_params:,} params "
        f"(in_ch={pe.in_channels}, dim={pe.out_channels}, "
        f"kernel={list(pe.kernel_size)}, stride={list(pe.stride)})"
    )
    return module
