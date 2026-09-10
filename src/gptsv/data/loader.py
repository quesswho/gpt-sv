"""Memory-mapped token shards and a deterministic, seekable batch loader.

A shard is a flat `.bin` file of token ids (documents concatenated, each one
terminated by `<|endoftext|>`), written by `gptsv-data tokenize`. Shards are
memory-mapped, so the page cache does the work and startup is instant at any
corpus size.
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

    Each shard is cut into non-overlapping blocks. Every epoch visits each
    block exactly once, in an order drawn from (seed, epoch), dealt round-robin
    across ranks - so no rank repeats or skips data within an epoch, unlike
    sampling random offsets, which leaves ~37% of the corpus unseen after one
    nominal pass.

    Batch `i` is a pure function of (seed, rank, world_size, i). That is what
    makes `seek` exact: a resumed run sees precisely the batches an
    uninterrupted run would have.
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
            # A batch can straddle an epoch boundary, so keep the previous
            # epoch's order around rather than regenerating it per sample.
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
