"""
Inference dataset for FlowWAM world model on RoboTwin / WorldArena.
"""

import os
import json
import glob
from typing import List, Dict, Optional, Tuple

import cv2
import h5py
import numpy as np
from PIL import Image
from io import BytesIO

from reversible_flow_codec import FlowCodec
from video_flow_codec_pipeline import RAFTFlowExtractor


INSTRUCTION_PREFIX = (
    "In a fixed robotic workspace, generate a rigid, physically consistent "
    "embodied robotic arm. The arm maintains high stability with no deformation "
    "and enters the frame to "
)


def _detect_bg_color(frame: np.ndarray) -> np.ndarray:
    """Return the most frequent pixel color in *frame* (H, W, 3) uint8."""
    pixels = frame.reshape(-1, 3)
    quantized = (pixels // 4) * 4
    keys = quantized[:, 0].astype(np.int32) * 65536 + quantized[:, 1].astype(np.int32) * 256 + quantized[:, 2].astype(np.int32)
    counts = np.bincount(keys)
    mode_key = counts.argmax()
    r = (mode_key // 65536) & 0xFF
    g = (mode_key // 256) & 0xFF
    b = mode_key & 0xFF
    return np.array([r, g, b], dtype=np.uint8)


def _make_static_texture(shape: Tuple[int, int, int], bg_color: np.ndarray,
                         seed: int = 0) -> np.ndarray:
    """Generate a deterministic static texture around *bg_color*."""
    rng = np.random.RandomState(seed)
    noise = rng.randint(-10, 11, size=shape, dtype=np.int16)
    tex = np.clip(bg_color.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return tex


def add_bg_texture(
    frames: List[np.ndarray],
    bg_tol: int = 10,
) -> List[np.ndarray]:
    """Replace solid background with a static texture in every frame."""
    bg_color = _detect_bg_color(frames[0])
    texture = _make_static_texture(frames[0].shape, bg_color, seed=42)
    batch = np.stack(frames, axis=0)
    diff = np.max(np.abs(batch.astype(np.int16) - bg_color.astype(np.int16)), axis=-1)
    is_bg = diff <= bg_tol
    out = batch.copy()
    bg3 = np.broadcast_to(is_bg[..., None], out.shape)
    tex_broadcast = np.broadcast_to(texture[None], out.shape)
    np.copyto(out, tex_broadcast, where=bg3)
    return [out[i] for i in range(len(frames))]


def mask_flows_by_robot(
    flows: List[np.ndarray],
    frames: List[np.ndarray],
    bg_tol: int = 10,
) -> List[np.ndarray]:
    """Keep flow only where the source frame contains the robot."""
    bg_color = _detect_bg_color(frames[0])
    masked = []
    for i, flow in enumerate(flows):
        diff_src = np.max(np.abs(frames[i].astype(np.int16) - bg_color.astype(np.int16)), axis=-1)
        is_robot = diff_src > bg_tol
        flow_clean = np.zeros_like(flow)
        flow_clean[is_robot] = flow[is_robot]
        masked.append(flow_clean)
    return masked


def compute_flow_farneback(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow


def _uniform_resample_indices(T: int, num_frames: int) -> List[int]:
    """Map [0, num_frames-1] -> [0, T-1] uniformly."""
    if T <= 1:
        return [0] * num_frames
    return [round(i * (T - 1) / (num_frames - 1)) for i in range(num_frames)]


class RoboTwinWorldModelInferenceDataset:
    """Inference dataset for WorldArena evaluation.

    Parameters
    ----------
    test_dataset_dir : str
        Path to WorldArena ``test_dataset/`` directory.
    robot_only_dir : str or None
        Pre-generated robot-only HDF5 directory, or None to render via SAPIEN.
    camera : str
        Camera name (default ``head_camera``).
    size : tuple of (width, height)
        Target output resolution.
    num_frames : int
        Output frame count (default 121).
    flow_method, flow_device, flow_max_magnitude
        Flow computation parameters.
    embodiment_dir : str or None
        Directory containing ``embodiments/`` (required when ``robot_only_dir`` is None).
    variant : str
        Robot variant for SAPIEN rendering.
    flow_resolution : tuple or None
        Resolution for flow computation (upsampled to ``size`` afterwards).
    robot_render_resolution : tuple or None
        Override SAPIEN camera resolution.
    """

    def __init__(
        self,
        test_dataset_dir: str,
        robot_only_dir: Optional[str] = None,
        camera: str = "head_camera",
        size: Tuple[int, int] = (640, 480),
        num_frames: int = 121,
        flow_method: str = "raft",
        flow_device: str = "cuda",
        flow_max_magnitude: Optional[float] = None,
        embodiment_dir: Optional[str] = None,
        variant: str = "aloha-agilex_clean_50",
        instruction_variant: int = 0,
        flow_resolution: Optional[Tuple[int, int]] = None,
        robot_render_resolution: Optional[Tuple[int, int]] = None,
    ):
        self.test_dataset_dir = test_dataset_dir
        self.robot_only_dir = robot_only_dir
        self.camera = camera
        self.size = size
        self.num_frames = num_frames
        self.flow_method = flow_method
        self.flow_device = flow_device
        self.flow_max_magnitude = flow_max_magnitude
        self.embodiment_dir = embodiment_dir
        self.variant = variant
        self.instruction_variant = instruction_variant
        self.flow_resolution = flow_resolution
        self.robot_render_resolution = robot_render_resolution

        self._use_sapien = (robot_only_dir is None)
        if self._use_sapien and embodiment_dir is None:
            raise ValueError(
                "embodiment_dir is required when robot_only_dir is not provided")

        self.codec = FlowCodec()
        self._flow_extractor = None
        self._robot_renderer = None

        self.samples = self._discover_episodes()
        mode = "SAPIEN on-the-fly" if self._use_sapien else "pre-generated HDF5"
        instr_label = "instructions" if instruction_variant == 0 else f"instructions_{instruction_variant}"
        flow_res_str = f"{flow_resolution[0]}x{flow_resolution[1]}" if flow_resolution else "native"
        print(f"[RoboTwinInference] {len(self.samples)} episodes, "
              f"size={size}, num_frames={num_frames}, "
              f"robot_only={mode}, instructions={instr_label}, "
              f"flow_resolution={flow_res_str}")

    def _discover_episodes(self) -> List[Dict]:
        first_frame_dir = os.path.join(
            self.test_dataset_dir, "first_frame", "fixed_scene_task"
        )
        instr_subdir = (
            "instructions" if self.instruction_variant == 0
            else f"instructions_{self.instruction_variant}"
        )
        instr_dir = os.path.join(
            self.test_dataset_dir, instr_subdir, "fixed_scene_task"
        )
        action_dir = os.path.join(
            self.test_dataset_dir, "data", "fixed_scene_task"
        )

        samples = []
        for png_path in sorted(glob.glob(os.path.join(first_frame_dir, "episode*.png"))):
            ep_name = os.path.splitext(os.path.basename(png_path))[0]
            instr_path = os.path.join(instr_dir, f"{ep_name}.json")
            action_path = os.path.join(action_dir, f"{ep_name}.hdf5")

            if not os.path.exists(action_path):
                continue

            sample = {
                "episode_name": ep_name,
                "first_frame_path": png_path,
                "instruction_path": instr_path,
                "action_hdf5": action_path,
                "robot_only_hdf5": None,
            }

            if self.robot_only_dir is not None:
                ro_path = os.path.join(self.robot_only_dir, f"{ep_name}.hdf5")
                if not os.path.exists(ro_path):
                    print(f"  Warning: robot_only not found for {ep_name}, skipping")
                    continue
                sample["robot_only_hdf5"] = ro_path

            samples.append(sample)

        return samples

    @staticmethod
    def _load_hdf5_rgb_frame_raw(hdf5_file, camera: str, idx: int) -> np.ndarray:
        jpeg_bytes = hdf5_file[f"observation/{camera}/rgb"][idx]
        img = Image.open(BytesIO(bytes(jpeg_bytes)))
        return np.array(img.convert("RGB"), dtype=np.uint8)

    def _resize_flow(self, flow: np.ndarray) -> np.ndarray:
        """Resize flow field to self.size with proper scaling."""
        h_orig, w_orig = flow.shape[:2]
        w_tgt, h_tgt = self.size
        if (w_orig, h_orig) == (w_tgt, h_tgt):
            return flow
        flow_resized = cv2.resize(flow, (w_tgt, h_tgt), interpolation=cv2.INTER_LINEAR)
        flow_resized[..., 0] *= w_tgt / w_orig
        flow_resized[..., 1] *= h_tgt / h_orig
        return flow_resized

    def _compute_flows(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if self.flow_method == "raft":
            if self._flow_extractor is None:
                self._flow_extractor = RAFTFlowExtractor(device=self.flow_device)
            return self._flow_extractor.batch_call(frames)
        return [compute_flow_farneback(frames[i], frames[i + 1])
                for i in range(len(frames) - 1)]

    def _encode_flows_to_pil(
        self, flows: List[np.ndarray]
    ) -> Tuple[List[Image.Image], List[float]]:
        pil_list = []
        max_mag_list = []
        for flow in flows:
            rgb, max_mag = self.codec.encode(flow, max_magnitude=self.flow_max_magnitude)
            pil_list.append(Image.fromarray(rgb))
            max_mag_list.append(float(max_mag))
        return pil_list, max_mag_list

    def __len__(self):
        return len(self.samples)

    def _ensure_robot_renderer(self):
        if self._robot_renderer is not None:
            return
        from robot_only_renderer import RobotOnlyRenderer
        self._robot_renderer = RobotOnlyRenderer(
            embodiment_dir=self.embodiment_dir,
            variant=self.variant,
            render_resolution=self.robot_render_resolution,
        )

    def _get_robot_only_frames_from_hdf5(self, sample: Dict) -> List[np.ndarray]:
        with h5py.File(sample["robot_only_hdf5"], "r") as f_ro:
            T_ro = f_ro[f"observation/{self.camera}/rgb"].shape[0]
            indices = _uniform_resample_indices(T_ro, self.num_frames)
            return [
                self._load_hdf5_rgb_frame_raw(f_ro, self.camera, t_idx)
                for t_idx in indices
            ]

    def _get_robot_only_frames_from_sapien(self, sample: Dict) -> List[np.ndarray]:
        self._ensure_robot_renderer()
        renderer = self._robot_renderer
        T = renderer.get_episode_length(sample["action_hdf5"])
        indices = _uniform_resample_indices(T, self.num_frames)
        return renderer.render_episode(
            sample["action_hdf5"], indices, camera=self.camera)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        first_frame = Image.open(sample["first_frame_path"]).convert("RGB")
        w, h = self.size
        first_frame = first_frame.resize((w, h), Image.BICUBIC)

        prompt = ""
        if os.path.exists(sample["instruction_path"]):
            with open(sample["instruction_path"], "r") as f:
                data = json.load(f)
            prompt = data.get("instruction", "")

        if sample["robot_only_hdf5"] is not None:
            robot_only_frames_native = self._get_robot_only_frames_from_hdf5(sample)
        else:
            robot_only_frames_native = self._get_robot_only_frames_from_sapien(sample)

        if self.flow_resolution is not None:
            fw, fh = self.flow_resolution
            robot_only_for_flow = [
                cv2.resize(f, (fw, fh), interpolation=cv2.INTER_LINEAR)
                for f in robot_only_frames_native
            ]
        else:
            robot_only_for_flow = robot_only_frames_native

        textured_frames = add_bg_texture(robot_only_for_flow)
        flows = self._compute_flows(textured_frames)
        flows = mask_flows_by_robot(flows, robot_only_for_flow)
        flows_resized = [self._resize_flow(f) for f in flows]
        flow_pil_list, max_mag_list = self._encode_flows_to_pil(flows_resized)

        zero_flow_pil = Image.fromarray(np.full((h, w, 3), 255, dtype=np.uint8))
        flow_pil_list.insert(0, zero_flow_pil)
        max_mag_list.insert(0, 0.0)

        return {
            "rgb_video": None,
            "flow_video": flow_pil_list,
            "reference_image": first_frame,
            "prompt": prompt,
            "max_magnitudes": max_mag_list,
            "episode_name": sample["episode_name"],
        }
