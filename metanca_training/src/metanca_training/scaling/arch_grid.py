"""Deterministic architecture grid for the scaling ablation.

An architecture is a tuple of hidden-layer widths, non-increasing, drawn from
powers of two in {32,64,128,256,512}. The output layer (n_classes) is appended
only at instantiation time.
"""

from itertools import combinations_with_replacement
from typing import Sequence

import jax
import numpy as np

from metanca.nn import MultiLayerPerceptron

WIDTHS: tuple[int, ...] = (16, 32, 64, 128)

# Held-out validation-architecture counts.
# fixed3: 20 archs total -> hold out 8, training pool 12 (T in {1,2,4,8[,12]}).
N_VAL: dict[str, int] = {"fixed3": 8, "varying": 24}

_GRID_DEPTHS: dict[str, list[int]] = {"fixed3": [3], "varying": [2, 3, 4, 5]}

# Grid-wide maxima used to provision the explicit grid-max hidden-state initializer
# (`grid_hidden_state_initializer`). The initializer's layer table must cover
# MAX_HIDDEN_LAYERS + 1 (output) positions and its neuron table max(input_dim, MAX_HIDDEN_WIDTH).
# Provisioned to the varying-grid max depth so both ablations share one provisioning
# (PE values are position-invariant; oversized tables are harmless).
MAX_HIDDEN_WIDTH: int = max(WIDTHS)                    # 128
MAX_HIDDEN_LAYERS: int = max(_GRID_DEPTHS["varying"])  # 5


def enumerate_arch_widths(
    depths: Sequence[int], widths: Sequence[int] = WIDTHS
) -> list[tuple[int, ...]]:
    """All non-increasing width tuples of the given depths.

    combinations_with_replacement over descending-sorted widths yields exactly
    the non-increasing tuples, one per multiset (no duplicates).
    """
    desc = tuple(sorted(widths, reverse=True))
    archs: list[tuple[int, ...]] = []
    for d in depths:
        archs.extend(combinations_with_replacement(desc, d))
    return archs


def build_grid(kind: str) -> list[tuple[int, ...]]:
    if kind not in _GRID_DEPTHS:
        raise ValueError(f"unknown grid kind: {kind!r}")
    return enumerate_arch_widths(_GRID_DEPTHS[kind])


def arch_id(widths: Sequence[int]) -> str:
    return f"d{len(widths)}_" + "-".join(str(w) for w in widths)


def arch_layer_specs(widths: Sequence[int], n_classes: int) -> list[int]:
    return [*widths, n_classes]


def build_mlp(
    widths: Sequence[int], n_classes: int, use_bias: bool = True
) -> MultiLayerPerceptron:
    return MultiLayerPerceptron(
        layer_specs=arch_layer_specs(widths, n_classes),
        activation="leaky_relu",
        final_layer_activation="identity",
        use_bias=use_bias,
    )


def _shuffled(grid: Sequence[tuple[int, ...]], seed: int) -> list[tuple[int, ...]]:
    order = np.array(jax.random.permutation(jax.random.key(seed), len(grid)))
    return [tuple(grid[i]) for i in order]


def split_grid(
    grid: Sequence[tuple[int, ...]], n_val: int, seed: int
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Return (pool, val). Deterministic in seed; disjoint; covers the grid."""
    shuffled = _shuffled(grid, seed)
    val = shuffled[:n_val]
    pool = shuffled[n_val:]
    return pool, val


def sample_subset(
    pool: Sequence[tuple[int, ...]], t: int, seed: int
) -> list[tuple[int, ...]]:
    if t > len(pool):
        raise ValueError(f"cannot sample T={t} from pool of {len(pool)}")
    return _shuffled(pool, seed)[:t]
