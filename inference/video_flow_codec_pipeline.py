"""
Optical flow extractors (RAFT / Farneback).
"""

import numpy as np
import cv2

from reversible_flow_codec import FlowCodec


# ============================================================
#  Pretrained Flow Extractors
# ============================================================

class RAFTFlowExtractor:
    """
    Dense optical flow via RAFT pretrained model (torchvision).
    """

    def __init__(self, device='cuda'):
        import torch
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(self.device).eval()

    def _preprocess(self, frame):
        """(H,W,3) uint8 -> (1,3,H',W') normalized float tensor, padded to 8x."""
        import torch
        import torch.nn.functional as F

        img = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0
        img = (img - 0.5) / 0.5
        img = img.unsqueeze(0)

        _, _, h, w = img.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='replicate')

        return img.to(self.device), h, w

    def __call__(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Compute dense optical flow from frame1 to frame2.

        Args:
            frame1, frame2: (H, W, 3) uint8 RGB.
        Returns:
            flow: (H, W, 2) float32, flow[...,0]=dx, flow[...,1]=dy in pixels.
        """
        import torch

        img1, orig_h, orig_w = self._preprocess(frame1)
        img2, _, _ = self._preprocess(frame2)

        with torch.no_grad(), torch.amp.autocast('cuda'):
            flow_preds = self.model(img1, img2)

        flow = flow_preds[-1].squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        return flow[:orig_h, :orig_w].astype(np.float32)

    def batch_call(self, frames: list, max_batch_size: int = 120,
                   use_autocast: bool = True) -> list:
        """
        Batch-compute optical flow for all consecutive frame pairs.

        Args:
            frames: List of (H, W, 3) uint8 RGB frames (length N).
            max_batch_size: Max pairs per forward pass. Default 120 fits
                120 pairs of 320x240 in ~1.5 GB VRAM with raft_large.
            use_autocast: If True, use fp16 autocast (fast but cuDNN may
                reject grid_sample at high resolutions). Set False for fp32.
        Returns:
            List of (H, W, 2) float32 flow arrays (length N-1).
        """
        import torch

        if len(frames) < 2:
            return []

        tensors = []
        orig_h, orig_w = None, None
        for frame in frames:
            t, h, w = self._preprocess(frame)
            tensors.append(t)
            if orig_h is None:
                orig_h, orig_w = h, w

        img1_all = torch.cat(tensors[:-1], dim=0)
        img2_all = torch.cat(tensors[1:], dim=0)
        n_pairs = img1_all.shape[0]

        flow_results = []
        for start in range(0, n_pairs, max_batch_size):
            end = min(start + max_batch_size, n_pairs)
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_autocast):
                preds = self.model(img1_all[start:end], img2_all[start:end])
            batch_flow = preds[-1].float().permute(0, 2, 3, 1).cpu().numpy()
            for i in range(batch_flow.shape[0]):
                flow_results.append(batch_flow[i, :orig_h, :orig_w].astype(np.float32))

        return flow_results


def compute_flow_farneback(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """Farneback dense optical flow (classical, no GPU required)."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=5, winsize=15,
        iterations=5, poly_n=7, poly_sigma=1.5, flags=0,
    )
    return flow.astype(np.float32)
