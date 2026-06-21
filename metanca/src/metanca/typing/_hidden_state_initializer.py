from typing import Callable, Sequence

import jax

HiddenStateInitializer = Callable[
    [Sequence[int], int, bool, float], tuple[jax.Array, jax.Array | None]
]
