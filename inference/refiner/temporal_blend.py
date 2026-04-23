"""Temporal blending utilities for the Stage-2 refiner.

final = alpha * refined + (1 - alpha) * original

Operates on numpy uint8 arrays in memory; no disk I/O.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def blend_arrays(
    refined: np.ndarray | Sequence[np.ndarray],
    original: np.ndarray | Sequence[np.ndarray],
    alpha: float = 0.7,
) -> np.ndarray:
    """Blend two clips (refined vs. original) frame-by-frame.

    Both inputs are converted to (T, H, W, C) uint8 ndarrays. The original
    clip is bicubic-resized to the refined resolution if shapes differ; the
    shorter clip determines the output length.
    """
    if isinstance(refined, (list, tuple)):
        refined = np.stack([np.asarray(f) for f in refined], axis=0)
    if isinstance(original, (list, tuple)):
        original = np.stack([np.asarray(f) for f in original], axis=0)

    refined = np.asarray(refined)
    original = np.asarray(original)

    if refined.ndim != 4 or original.ndim != 4:
        raise ValueError(
            f"expect (T,H,W,C) arrays, got refined={refined.shape} "
            f"original={original.shape}"
        )

    n = min(refined.shape[0], original.shape[0])
    refined = refined[:n]
    original = original[:n]

    if original.shape[1:] != refined.shape[1:]:
        h, w = refined.shape[1], refined.shape[2]
        original = np.stack(
            [cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC) for f in original],
            axis=0,
        )

    blended = (
        alpha * refined.astype(np.float32)
        + (1.0 - alpha) * original.astype(np.float32)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)
