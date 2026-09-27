"""Memory-mapped token shards and a deterministic, seekable batch loader.

A shard is a flat `.bin` file of token ids (documents concatenated, each one
terminated by `<|endoftext|>`), written by `gptsv-data tokenize`. Shards are
memory-mapped, so startup is fast at any corpus size.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class ShardDataset:
    """Read-only view over every `*.bin` shard in a directory."""

    def __init__(self, directory: str | Path, dtype: str = "uint16"):
        self.directory = Path(directory)
        self.paths = sorted(self.directory.glob("*.bin"))
        if not self.paths:
            raise FileNotFoundError(f"no *.bin shards in {self.directory}")
        self.dtype = np.dtype(dtype)
        self.shards = [np.memmap(p, dtype=self.dtype, mode="r") for p in self.paths]

    def __len__(self) -> int:
        return sum(len(s) for s in self.shards)

    def __repr__(self) -> str:
        return f"ShardDataset({self.directory}, {len(self.paths)} shards, {len(self):,} tokens)"


class TokenLoader:
    """Yields [batch_size, block_len] int64 batches of non-overlapping token blocks.

    Each epoch visits every block exactly once, in an order drawn from
    (seed, epoch) and dealt round-robin across ranks.

    Batch `i` depends only on (seed, rank, world_size, i), so after `seek` a
    resumed run sees the same batches an uninterrupted run would have.
    """

    def __init__(
        self,
        dataset: ShardDataset,
        batch_size: int,
        block_len: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ):
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} out of range for world_size {world_size}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.block_len = block_len
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        counts = [len(s) // block_len for s in dataset.shards]
        self._block_shard = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
        self._block_start = np.concatenate([np.arange(c, dtype=np.int64) * block_len for c in counts])
        self.n_blocks = int(sum(counts))
        self.blocks_per_rank = self.n_blocks // world_size
        if self.blocks_per_rank < batch_size:
            raise ValueError(
                f"{dataset} holds {self.n_blocks} blocks of {block_len} tokens; "
                f"too few for batch_size {batch_size} across {world_size} ranks"
            )

        self._batch = 0
        self._perms: dict[int, np.ndarray] = {}

    def seek(self, batch_index: int) -> None:
        """Position the loader so the next batch is batch number `batch_index`."""
        self._batch = batch_index

    def _perm(self, epoch: int) -> np.ndarray:
        if epoch not in self._perms:
            # Keep the previous epoch's order for batches that span two epochs.
            self._perms = {e: p for e, p in self._perms.items() if e == epoch - 1}
            rng = np.random.default_rng([self.seed, epoch])
            self._perms[epoch] = rng.permutation(self.n_blocks)
        return self._perms[epoch]

    def __iter__(self) -> TokenLoader:
        return self

    def __next__(self) -> torch.Tensor:
        out = np.empty((self.batch_size, self.block_len), dtype=np.int64)
        first = self._batch * self.batch_size
        for j in range(self.batch_size):
            epoch, pos = divmod(first + j, self.blocks_per_rank)
            block = self._perm(epoch)[pos * self.world_size + self.rank]
            shard = self.dataset.shards[self._block_shard[block]]
            start = self._block_start[block]
            out[j] = shard[start : start + self.block_len]
        self._batch += 1
        return torch.from_numpy(out)
