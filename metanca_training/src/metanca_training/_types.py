from typing import Callable, TypeAlias

import chex
import jax

UpdateFn = Callable[[jax.Array, jax.Array, chex.PRNGKey], tuple[jax.Array, jax.Array]]
ApplyFn = Callable[[chex.ArrayTree, jax.Array], jax.Array]
Device: TypeAlias = object
