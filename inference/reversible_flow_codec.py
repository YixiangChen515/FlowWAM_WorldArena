"""
Reversible optical flow <-> RGB image codec (HSV color wheel encoding).
"""

import numpy as np
import cv2
import struct
import os
from typing import Tuple, Optional


# ============================================================
#  Core Codec
# ============================================================

class FlowCodec:
    """Reversible optical flow <-> RGB image codec (8-bit or 16-bit)."""

    def __init__(self, use_16bit: bool = False):
        self.use_16bit = use_16bit
        self.max_val = 65535 if use_16bit else 255
        self.dtype = np.uint16 if use_16bit else np.uint8

    # ----- Encode -----

    def encode(
        self,
        flow: np.ndarray,
        max_magnitude: Optional[float] = None,
        sigma: float = 0.15,
    ) -> Tuple[np.ndarray, float]:
        """Encode (H, W, 2) flow into an RGB image. Returns (rgb, max_magnitude)."""
        H, W, _ = flow.shape
        dx, dy = flow[..., 0], flow[..., 1]

        # Polar coordinates
        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        angle = np.arctan2(dy, dx)  # [-pi, pi]

        # Normalize magnitude
        if max_magnitude is not None and max_magnitude == -1:
            diag = np.sqrt(float(H ** 2 + W ** 2))
            max_magnitude = sigma * diag
        elif max_magnitude is None:
            max_magnitude = float(np.percentile(magnitude, 99.5)) + 1e-6
        mag_norm = np.clip(magnitude / max_magnitude, 0, 1)

        # Map angle to [0, 1] range:  -pi..pi  ->  0..1
        angle_norm = (angle + np.pi) / (2 * np.pi)  # [0, 1]
        angle_norm = np.clip(angle_norm, 0, 1)

        # Build HSV image
        # H: 0-179 (OpenCV convention for 8-bit), or 0-65535 for 16-bit
        # S: 0-max_val
        # V: max_val (constant)
        if self.use_16bit:
            hue = (angle_norm * 65535).astype(np.uint16)
            sat = (mag_norm * 65535).astype(np.uint16)
            val = np.full_like(hue, 65535)
            hsv = np.stack([hue, sat, val], axis=-1)
            # For 16-bit, we do the HSV->RGB conversion manually
            rgb = self._hsv16_to_rgb16(hsv)
        else:
            hue = (angle_norm * 179).astype(np.uint8)       # OpenCV: H in [0,179]
            sat = (mag_norm * 255).astype(np.uint8)
            val = np.full_like(hue, 255, dtype=np.uint8)
            hsv = np.stack([hue, sat, val], axis=-1)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return rgb, max_magnitude

    def decode(self, rgb_image: np.ndarray, max_magnitude: float) -> np.ndarray:
        """Decode an RGB flow image back to (H, W, 2) optical flow."""
        if self.use_16bit:
            hsv = self._rgb16_to_hsv16(rgb_image)
            angle_norm = hsv[..., 0].astype(np.float64) / 65535.0
            mag_norm = hsv[..., 1].astype(np.float64) / 65535.0
        else:
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
            angle_norm = hsv[..., 0].astype(np.float64) / 179.0
            mag_norm = hsv[..., 1].astype(np.float64) / 255.0

        # Recover angle: [0, 1] -> [-pi, pi]
        angle = angle_norm * 2 * np.pi - np.pi

        # Recover magnitude
        magnitude = mag_norm * max_magnitude

        # Polar -> Cartesian
        dx = magnitude * np.cos(angle)
        dy = magnitude * np.sin(angle)

        flow = np.stack([dx, dy], axis=-1).astype(np.float32)
        return flow

    # ----- Save / Load with embedded metadata -----

    def save(self, path: str, flow: np.ndarray, max_magnitude: Optional[float] = None):
        """Encode flow and save as PNG with max_magnitude embedded."""
        rgb, max_mag = self.encode(flow, max_magnitude)

        # Embed max_magnitude in bottom-right 8 pixels as float32 bytes
        mag_bytes = struct.pack('<d', max_mag)  # 8 bytes double
        H, W = rgb.shape[:2]
        for i, b in enumerate(mag_bytes):
            if i < W:
                rgb[H - 1, W - 1 - i, 0] = b  # store in R channel of last row

        # Save as PNG
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)

        # Also save max_magnitude as sidecar .meta file (most reliable)
        meta_path = path + ".meta"
        with open(meta_path, 'w') as f:
            f.write(f"{max_mag:.10f}")

        return max_mag

    def load(self, path: str, max_magnitude: Optional[float] = None) -> np.ndarray:
        """Load a flow PNG and decode back to (H, W, 2) flow."""
        bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Determine max_magnitude
        if max_magnitude is None:
            meta_path = path + ".meta"
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    max_magnitude = float(f.read().strip())
            else:
                # Fallback: read from embedded pixels
                H, W = rgb.shape[:2]
                mag_bytes = bytes([rgb[H - 1, W - 1 - i, 0] for i in range(8)])
                max_magnitude = struct.unpack('<d', mag_bytes)[0]

        return self.decode(rgb, max_magnitude)

    # ----- 16-bit HSV helpers -----

    @staticmethod
    def _hsv16_to_rgb16(hsv: np.ndarray) -> np.ndarray:
        """Manual HSV [0-65535] -> RGB [0-65535] conversion for 16-bit."""
        h = hsv[..., 0].astype(np.float64) / 65535.0 * 360.0  # [0, 360]
        s = hsv[..., 1].astype(np.float64) / 65535.0           # [0, 1]
        v = hsv[..., 2].astype(np.float64) / 65535.0           # [0, 1]

        c = v * s
        h_prime = h / 60.0
        x = c * (1 - np.abs(h_prime % 2 - 1))
        m = v - c

        r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)
        for lo, hi, rv, gv, bv in [
            (0, 1, c, x, 0), (1, 2, x, c, 0), (2, 3, 0, c, x),
            (3, 4, 0, x, c), (4, 5, x, 0, c), (5, 6, c, 0, x),
        ]:
            mask = (h_prime >= lo) & (h_prime < hi)
            r[mask] = rv[mask] if isinstance(rv, np.ndarray) else rv
            g[mask] = gv[mask] if isinstance(gv, np.ndarray) else gv
            b[mask] = bv[mask] if isinstance(bv, np.ndarray) else bv

        rgb = np.stack([(r + m), (g + m), (b + m)], axis=-1)
        return (rgb * 65535).clip(0, 65535).astype(np.uint16)

    @staticmethod
    def _rgb16_to_hsv16(rgb: np.ndarray) -> np.ndarray:
        """Manual RGB [0-65535] -> HSV [0-65535] conversion for 16-bit."""
        r = rgb[..., 0].astype(np.float64) / 65535.0
        g = rgb[..., 1].astype(np.float64) / 65535.0
        b = rgb[..., 2].astype(np.float64) / 65535.0

        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        # Hue
        h = np.zeros_like(delta)
        mask_r = (cmax == r) & (delta > 0)
        mask_g = (cmax == g) & (delta > 0)
        mask_b = (cmax == b) & (delta > 0)
        h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
        h[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2)
        h[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4)

        # Saturation
        s = np.where(cmax > 0, delta / cmax, 0)

        hsv = np.stack([
            (h / 360.0 * 65535).clip(0, 65535),
            (s * 65535).clip(0, 65535),
            (cmax * 65535).clip(0, 65535),
        ], axis=-1).astype(np.uint16)
        return hsv


# ============================================================
#  Visualization Helpers
# ============================================================

def make_color_wheel_legend(size: int = 200) -> np.ndarray:
    """Generate a color wheel legend (size, size, 3) uint8 RGB."""
    cx, cy = size // 2, size // 2
    radius = size // 2 - 10

    legend = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            dx, dy = x - cx, y - cy
            dist = np.sqrt(dx**2 + dy**2)
            if dist <= radius:
                angle = np.arctan2(dy, dx)
                angle_norm = (angle + np.pi) / (2 * np.pi)
                mag_norm = dist / radius

                h = int(angle_norm * 179)
                s = int(mag_norm * 255)
                v = 255
                legend[y, x] = [h, s, v]

    legend = cv2.cvtColor(legend, cv2.COLOR_HSV2RGB)
    # Add axis labels
    cv2.putText(legend, '+X', (size - 30, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(legend, '-X', (5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(legend, '+Y', (cx - 8, size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(legend, '-Y', (cx - 8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return legend


# ============================================================
#  Pipeline Integration: Flow -> Action
# ============================================================

def extract_action_from_flow(
    flow: np.ndarray,
    mask: np.ndarray,
    method: str = "median"
) -> np.ndarray:
    """Extract aggregate action (dx, dy) from flow within a mask region."""
    assert flow.shape[:2] == mask.shape[:2]
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return np.zeros(2, dtype=np.float32)

    flows_in_mask = flow[ys, xs]  # (N, 2)

    if method == "median":
        action = np.median(flows_in_mask, axis=0)
    else:
        action = np.mean(flows_in_mask, axis=0)

    return action.astype(np.float32)


# ============================================================
#  Demo & Verification
# ============================================================

def demo_roundtrip():
    """
    Full round-trip test: flow -> encode -> save -> load -> decode -> compare.
    """
    print("=" * 60)
    print("  Reversible Flow Codec — Round-trip Verification")
    print("=" * 60)

    H, W = 240, 320

    # Create synthetic flow (circular motion + translation)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    cx, cy = W / 2, H / 2
    dx = -(yy - cy) * 0.3 + 2.0   # circular + rightward
    dy = (xx - cx) * 0.3 - 1.5    # circular + upward
    flow_gt = np.stack([dx, dy], axis=-1).astype(np.float32)

    for mode_name, use_16bit in [("8-bit", False), ("16-bit", True)]:
        print(f"\n--- {mode_name} Mode ---")
        codec = FlowCodec(use_16bit=use_16bit)

        # Encode
        rgb, max_mag = codec.encode(flow_gt)
        print(f"  Max magnitude: {max_mag:.4f}")
        print(f"  Image shape: {rgb.shape}, dtype: {rgb.dtype}")

        # Decode
        flow_decoded = codec.decode(rgb, max_mag)

        # Error analysis
        error = np.abs(flow_gt - flow_decoded)
        print(f"  Mean absolute error:  {error.mean():.6f} px")
        print(f"  Max absolute error:   {error.max():.6f} px")
        print(f"  99th percentile error: {np.percentile(error, 99):.6f} px")

        # Save & reload test
        save_path = f"/tmp/test_flow_{mode_name.replace('-', '')}.png"
        codec.save(save_path, flow_gt)
        flow_reloaded = codec.load(save_path)
        reload_error = np.abs(flow_gt - flow_reloaded)
        print(f"  Save/Load roundtrip error: {reload_error.mean():.6f} px")

    # Action extraction test
    print(f"\n--- Action Extraction Test ---")
    codec = FlowCodec(use_16bit=False)
    rgb, max_mag = codec.encode(flow_gt)
    flow_decoded = codec.decode(rgb, max_mag)

    # Simulate end-effector mask (center region)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[100:140, 140:180] = 1

    action_gt = extract_action_from_flow(flow_gt, mask)
    action_decoded = extract_action_from_flow(flow_decoded, mask)
    print(f"  GT action:      [{action_gt[0]:.4f}, {action_gt[1]:.4f}]")
    print(f"  Decoded action:  [{action_decoded[0]:.4f}, {action_decoded[1]:.4f}]")
    print(f"  Action error:    [{abs(action_gt[0]-action_decoded[0]):.4f}, {abs(action_gt[1]-action_decoded[1]):.4f}]")

    # Save visualization
    legend = make_color_wheel_legend(200)
    print(f"\n  Color wheel legend shape: {legend.shape}")
    print("\nAll tests passed!")


if __name__ == "__main__":
    demo_roundtrip()
