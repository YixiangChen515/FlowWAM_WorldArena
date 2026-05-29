"""
Dataset for stride-capped + autoregressive dual-stream world model training.

Each episode is split into one or more rollout chunks of ``chunk_output_frames``
output frames. Short episodes get a single chunk with a small inter-frame
stride; long episodes get multiple chunks covering successive segments.

Cross-resolution loading:
  - Supervision frames + robot-only are read at the high-res training size.
  - The conditioning reference frame is read from the low-res dataset and
    BICUBIC-upsampled, matching the degradation seen at inference time.
Flow is computed at 320x240 via RAFT, then upscaled to the target size.
"""

import os
import json
import glob
import math
import random
from typing import List, Dict, Optional, Tuple, Union

import cv2
import h5py
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm

from reversible_flow_codec import FlowCodec
from video_flow_codec_pipeline import RAFTFlowExtractor
from dataset_world_robotwin import (
    INSTRUCTION_PREFIX,
    ROBOTWIN_ALL_TASKS,
    add_bg_texture,
    mask_flows_by_robot,
    compute_flow_farneback,
    _uniform_resample_indices,
)


class RoboTwinWorldModelDataset:
    """Training dataset with stride-capped subsampling and dynamic rollout count.

    Parameters
    ----------
    data_root : str
        Path to the high-res HDF5 data (supervision frames + robot-only).
    low_res_data_root : str
        Path to the low-res HDF5 data (conditioning first frame).
    chunk_output_frames : int
        Output frames per rollout chunk (must satisfy Wan VAE 4N+1, default 121).
    max_stride : int
        Maximum inter-frame stride within a single chunk (default 3).
    max_rollouts : int
        Maximum number of AR rollout chunks (default 2).
    """

    def __init__(
        self,
        data_root: str,
        low_res_data_root: str,
        variant: Union[str, List[str]] = "aloha-agilex_clean_50",
        camera: str = "head_camera",
        size: Tuple[int, int] = (640, 480),
        chunk_output_frames: int = 121,
        max_stride: int = 3,
        max_rollouts: int = 2,
        task_names: Optional[List[str]] = None,
        flow_method: str = "raft",
        flow_device: str = "cuda",
        flow_max_magnitude: Optional[float] = 25.0,
    ):
        self.data_root = data_root
        self.low_res_data_root = low_res_data_root
        # Normalize to a list of variants so downstream code can always do
        # `for v in self.variants`. Single-string callers (the older
        # train scripts) still work transparently because we accept str
        # here. self.variant is kept as the *first* variant for legacy
        # log strings; new code should use self.variants instead.
        if isinstance(variant, str):
            self.variants: List[str] = [variant]
        else:
            self.variants = list(variant)
        if len(self.variants) == 0:
            raise ValueError("variant must be a non-empty string or list of strings")
        self.variant = self.variants[0]
        self.camera = camera
        self.size = size
        self.chunk_output_frames = chunk_output_frames
        self.max_stride = max_stride
        self.max_rollouts = max_rollouts
        self.flow_method = flow_method
        self.flow_device = flow_device
        self.flow_max_magnitude = flow_max_magnitude
        self.load_from_cache = False

        self.codec = FlowCodec()
        self._flow_extractor = None

        if task_names is None or len(task_names) == 0:
            task_names = ROBOTWIN_ALL_TASKS
        self.task_names = task_names

        self.samples = self._discover_episodes()
        self.episode_lengths, self.rollout_counts = self._precompute_rollouts()

        # Bucket sample indices by num_rollouts. __getitem__'s fault-tolerance
        # path picks a substitute *from the same bucket* on failure so all DDP
        # ranks keep stepping with the same number of forward passes per sample;
        # mixing buckets within a single step would deadlock DDP grad sync.
        self._rollout_buckets: Dict[int, List[int]] = {}
        for i, n in enumerate(self.rollout_counts):
            self._rollout_buckets.setdefault(n, []).append(i)

        n_bucket1 = sum(1 for r in self.rollout_counts if r == 1)
        n_bucket2 = sum(1 for r in self.rollout_counts if r >= 2)
        variants_str = ",".join(self.variants)
        print(
            f"[Dataset] {len(self.samples)} episodes from "
            f"{len(set(s['task'] for s in self.samples))} tasks, "
            f"variants=[{variants_str}], camera={camera}, size={size}"
        )
        print(
            f"[Dataset] chunk={chunk_output_frames}, max_stride={max_stride}, "
            f"max_rollouts={max_rollouts}"
        )
        print(
            f"[Dataset] Bucket-1 (1 rollout): {n_bucket1}, "
            f"Bucket-2 (2 rollouts): {n_bucket2}"
        )
        print(f"[Dataset] hi-res root: {data_root}")
        print(f"[Dataset] lo-res root: {low_res_data_root}")

    def _discover_episodes(self) -> List[Dict]:
        samples = []
        # Multi-variant scan: each (task, variant) cell that has the full
        # tri-stream (hi-res data + hi-res robot_only + lo-res data) on
        # disk contributes its episodes to the flat sample list. Missing
        # cells are silently skipped, so it's OK if not every task has
        # every variant (e.g. new variants partway through generation).
        for task in sorted(self.task_names):
            for v in self.variants:
                hi_variant = os.path.join(self.data_root, task, v)
                hi_data = os.path.join(hi_variant, "data")
                hi_robot_only = os.path.join(hi_variant, "robot_only", "data")

                lo_variant = os.path.join(self.low_res_data_root, task, v)
                lo_data = os.path.join(lo_variant, "data")

                if not os.path.isdir(hi_data):
                    continue
                if not os.path.isdir(hi_robot_only):
                    continue
                if not os.path.isdir(lo_data):
                    continue

                hdf5_files = sorted(glob.glob(os.path.join(hi_data, "episode*.hdf5")))
                for hdf5_path in hdf5_files:
                    ep_name = os.path.splitext(os.path.basename(hdf5_path))[0]
                    ro_path = os.path.join(hi_robot_only, f"{ep_name}.hdf5")
                    lo_path = os.path.join(lo_data, f"{ep_name}.hdf5")

                    if not os.path.exists(ro_path) or not os.path.exists(lo_path):
                        continue

                    samples.append({
                        "task": task,
                        "variant": v,
                        "episode_name": ep_name,
                        "data_hdf5": hdf5_path,
                        "robot_only_hdf5": ro_path,
                        "low_res_hdf5": lo_path,
                        "variant_dir": hi_variant,
                    })

        if not samples:
            raise ValueError(
                f"No valid episodes found in {self.data_root} "
                f"for variants={self.variants}"
            )
        return samples

    def _precompute_rollouts(self):
        chunk_interval = (self.chunk_output_frames - 1) * self.max_stride
        lengths = []
        rollout_counts = []
        for sample in tqdm(self.samples, desc="Precomputing rollout counts"):
            with h5py.File(sample["data_hdf5"], "r") as f:
                T = f[f"observation/{self.camera}/rgb"].shape[0]
            lengths.append(T)
            n = min(self.max_rollouts, max(1, math.ceil((T - 1) / chunk_interval)))
            rollout_counts.append(n)
        return lengths, rollout_counts

    @staticmethod
    def _load_hdf5_rgb_frame_native(hdf5_file, camera: str, idx: int) -> np.ndarray:
        jpeg_bytes = hdf5_file[f"observation/{camera}/rgb"][idx]
        img = Image.open(BytesIO(bytes(jpeg_bytes)))
        return np.array(img.convert("RGB"), dtype=np.uint8)

    def _load_reference_frame(self, hdf5_file, camera: str, idx: int) -> np.ndarray:
        jpeg_bytes = hdf5_file[f"observation/{camera}/rgb"][idx]
        img = Image.open(BytesIO(bytes(jpeg_bytes)))
        w, h = self.size
        img = img.resize((w, h), Image.BICUBIC)
        return np.array(img, dtype=np.uint8)

    def _compute_flows(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if self.flow_method == "raft":
            if self._flow_extractor is None:
                self._flow_extractor = RAFTFlowExtractor(device=self.flow_device)
            return self._flow_extractor.batch_call(frames)
        return [compute_flow_farneback(frames[i], frames[i + 1])
                for i in range(len(frames) - 1)]

    def _resize_flow(self, flow: np.ndarray) -> np.ndarray:
        h_orig, w_orig = flow.shape[:2]
        w_tgt, h_tgt = self.size
        if (w_orig, h_orig) == (w_tgt, h_tgt):
            return flow
        flow_resized = cv2.resize(flow, (w_tgt, h_tgt), interpolation=cv2.INTER_LINEAR)
        flow_resized[..., 0] *= w_tgt / w_orig
        flow_resized[..., 1] *= h_tgt / h_orig
        return flow_resized

    def _encode_flows_to_pil(
        self, flows: List[np.ndarray]
    ) -> List[Image.Image]:
        pil_list = []
        for flow in flows:
            rgb, _ = self.codec.encode(flow, max_magnitude=self.flow_max_magnitude)
            pil_list.append(Image.fromarray(rgb))
        return pil_list

    def _get_prompt(self, sample: Dict) -> str:
        instr_dir = os.path.join(sample["variant_dir"], "instructions")
        instr_file = os.path.join(instr_dir, f"{sample['episode_name']}.json")
        task_desc = sample["task"].replace("_", " ")
        if os.path.exists(instr_file):
            with open(instr_file, "r") as f:
                data = json.load(f)
            if "seen" in data and len(data["seen"]) > 0:
                task_desc = random.choice(data["seen"])
        return INSTRUCTION_PREFIX + task_desc

    def _build_flat_indices(self, T: int, num_rollouts: int) -> List[int]:
        """Build a flat list of original-frame indices across all rollout chunks.

        Chunks overlap by 1 frame at boundaries.  Total length =
        (chunk_output_frames - 1) * num_rollouts + 1.
        """
        intervals_per_chunk = math.ceil((T - 1) / num_rollouts)
        all_indices: List[int] = []

        for chunk_i in range(num_rollouts):
            chunk_start = chunk_i * intervals_per_chunk
            chunk_end = min(chunk_start + intervals_per_chunk, T - 1)
            segment_len = chunk_end - chunk_start + 1

            local_indices = _uniform_resample_indices(
                segment_len, self.chunk_output_frames
            )
            global_indices = [chunk_start + li for li in local_indices]

            if chunk_i == 0:
                all_indices.extend(global_indices)
            else:
                all_indices.extend(global_indices[1:])

        return all_indices

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Fault-tolerant loader.

        If ``_load_sample(idx)`` raises (corrupt hdf5, broken jpeg, RAFT
        failure, ...), log a warning and retry with a random substitute
        drawn from the same ``num_rollouts`` bucket -- staying within the
        bucket is required so DDP ranks keep doing the same number of
        forward passes per sample, otherwise grad allreduce deadlocks.
        Capped at 8 attempts in case a whole bucket happens to be dead.
        """
        original_idx = idx
        bucket_key = self.rollout_counts[original_idx]
        attempts: List[str] = []
        for _ in range(8):
            try:
                return self._load_sample(idx)
            except Exception as e:
                s = self.samples[idx]
                attempts.append(
                    f"  idx={idx} {s['task']}/{s['variant']}/{s['episode_name']}: {e!r}"
                )
                print(
                    f"[Dataset] FAILED loading idx={idx} "
                    f"({s['task']}/{s['variant']}/{s['episode_name']}): {e!r}. "
                    f"Substituting from rollout-bucket {bucket_key}.",
                    flush=True,
                )
                idx = random.choice(self._rollout_buckets[bucket_key])
        raise RuntimeError(
            f"[Dataset] exhausted 8 fallback attempts starting at "
            f"idx={original_idx} (bucket={bucket_key}):\n"
            + "\n".join(attempts)
        )

    def _load_sample(self, idx: int) -> Dict:
        sample = self.samples[idx]
        T = self.episode_lengths[idx]
        num_rollouts = self.rollout_counts[idx]

        flat_indices = self._build_flat_indices(T, num_rollouts)

        with h5py.File(sample["data_hdf5"], "r") as f_hi, \
             h5py.File(sample["robot_only_hdf5"], "r") as f_ro, \
             h5py.File(sample["low_res_hdf5"], "r") as f_lo:

            rgb_frames = [
                self._load_hdf5_rgb_frame_native(f_hi, self.camera, i)
                for i in flat_indices
            ]
            robot_only_frames = [
                self._load_hdf5_rgb_frame_native(f_ro, self.camera, i)
                for i in flat_indices
            ]

            ref_frame = self._load_reference_frame(
                f_lo, self.camera, flat_indices[0]
            )

        FLOW_W, FLOW_H = 320, 240
        ro_small = [
            cv2.resize(f, (FLOW_W, FLOW_H), interpolation=cv2.INTER_LINEAR)
            for f in robot_only_frames
        ]
        textured_small = add_bg_texture(ro_small)
        flows = self._compute_flows(textured_small)
        flows = mask_flows_by_robot(flows, ro_small)
        flows = [self._resize_flow(f) for f in flows]
        flow_pil_list = self._encode_flows_to_pil(flows)

        w, h = self.size
        zero_flow_pil = Image.fromarray(np.full((h, w, 3), 255, dtype=np.uint8))
        flow_pil_list.insert(0, zero_flow_pil)

        rgb_video = [Image.fromarray(f) for f in rgb_frames]
        reference_image = Image.fromarray(ref_frame)
        prompt = self._get_prompt(sample)

        return {
            "rgb_video": rgb_video,
            "flow_video": flow_pil_list,
            "reference_image": reference_image,
            "prompt": prompt,
            "num_rollouts": num_rollouts,
        }
