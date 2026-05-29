"""
DDP-safe bucket sampler for dual-stream world model training.

Ensures all ranks process the same rollout count at every training step
by grouping samples into rollout-based buckets and sharding complete
blocks of ``num_replicas`` same-bucket samples across ranks.

Bucket-2 (multi-rollout) samples can be oversampled to increase the
proportion of AR training signal.
"""

import math
from typing import Iterator, List, Optional

import torch
from torch.utils.data import Sampler


class RolloutAwareSampler(Sampler[int]):
    """Bucket sampler that guarantees same-rollout batches across DDP ranks.

    Parameters
    ----------
    rollout_counts : list[int]
        Per-sample rollout count (1 or 2), from the dataset.
    num_replicas : int
        Total number of DDP processes (world size).
    rank : int
        Rank of the current process.
    bucket_oversample : int
        Repeat factor for Bucket-2 (rollout >= 2) samples.
        Increases AR training signal from ~12% to ~30%+ with oversample=3.
    seed : int
        Base random seed for reproducibility.
    drop_last : bool
        If True, drop incomplete blocks at the end of each bucket.
    """

    def __init__(
        self,
        rollout_counts: List[int],
        num_replicas: int,
        rank: int,
        bucket_oversample: int = 3,
        seed: int = 42,
        drop_last: bool = False,
    ):
        self.rollout_counts = rollout_counts
        self.num_replicas = num_replicas
        self.rank = rank
        self.bucket_oversample = bucket_oversample
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self.bucket_1 = [i for i, r in enumerate(rollout_counts) if r == 1]
        self.bucket_2 = [i for i, r in enumerate(rollout_counts) if r >= 2]

        bucket_2_effective = len(self.bucket_2) * bucket_oversample

        padded_1 = math.ceil(len(self.bucket_1) / num_replicas) * num_replicas
        padded_2 = math.ceil(bucket_2_effective / num_replicas) * num_replicas
        self.total_size = padded_1 + padded_2
        self.num_samples = self.total_size // num_replicas

    def _pad_to_block(self, indices: List[int]) -> List[int]:
        remainder = len(indices) % self.num_replicas
        if remainder > 0:
            pad = self.num_replicas - remainder
            indices = indices + indices[:pad]
        return indices

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        perm1 = torch.randperm(len(self.bucket_1), generator=g).tolist()
        shuffled_1 = [self.bucket_1[i] for i in perm1]

        bucket_2_expanded = self.bucket_2 * self.bucket_oversample
        perm2 = torch.randperm(len(bucket_2_expanded), generator=g).tolist()
        shuffled_2 = [bucket_2_expanded[i] for i in perm2]

        shuffled_1 = self._pad_to_block(shuffled_1)
        shuffled_2 = self._pad_to_block(shuffled_2)

        blocks: List[List[int]] = []
        nr = self.num_replicas
        for start in range(0, len(shuffled_1), nr):
            blocks.append(shuffled_1[start : start + nr])
        for start in range(0, len(shuffled_2), nr):
            blocks.append(shuffled_2[start : start + nr])

        block_perm = torch.randperm(len(blocks), generator=g).tolist()
        blocks = [blocks[i] for i in block_perm]

        indices: List[int] = []
        for block in blocks:
            indices.extend(block)

        assert len(indices) == self.total_size, (
            f"Expected {self.total_size} indices, got {len(indices)}"
        )

        indices = indices[self.rank :: self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
