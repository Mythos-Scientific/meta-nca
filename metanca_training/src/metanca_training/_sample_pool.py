"""Sample pool for maintaining intermediate NCA states across metaepochs.

Implements the sample pooling strategy from Growing Neural Cellular Automata
(Mordvintsev et al. 2020): maintain a pool of (params, hidden_states) snapshots,
sample from it instead of always resetting to random initialization.
"""

import random
from dataclasses import dataclass, field

import jax
import numpy as np


@dataclass
class SamplePool:
    """Ring-buffer pool of (params, hidden_states) snapshots for one architecture.

    Entries are stored as numpy arrays in host RAM (not GPU memory).
    """

    capacity: int
    params_pool: list[dict] = field(default_factory=list)
    hidden_pool: list[dict] = field(default_factory=list)
    _write_idx: int = 0

    @property
    def size(self) -> int:
        return len(self.params_pool)

    def add(self, params: dict, hidden_states: dict) -> None:
        """Add a snapshot to the pool, overwriting oldest if full."""
        if self.size < self.capacity:
            self.params_pool.append(params)
            self.hidden_pool.append(hidden_states)
        else:
            self.params_pool[self._write_idx] = params
            self.hidden_pool[self._write_idx] = hidden_states
        self._write_idx = (self._write_idx + 1) % self.capacity

    def sample(self) -> tuple[dict, dict]:
        """Uniformly sample a (params, hidden_states) pair from the pool."""
        idx = random.randint(0, self.size - 1)
        return self.params_pool[idx], self.hidden_pool[idx]

    def clear(self) -> None:
        """Remove all entries from the pool."""
        self.params_pool.clear()
        self.hidden_pool.clear()
        self._write_idx = 0

    @staticmethod
    def to_numpy(tree: dict) -> dict:
        """Move a JAX pytree to host RAM as numpy arrays."""
        return jax.tree.map(np.asarray, jax.device_get(tree))

    @staticmethod
    def to_jax(tree: dict) -> dict:
        """Move a numpy pytree back to the default JAX device."""
        return jax.tree.map(jax.numpy.array, tree)
