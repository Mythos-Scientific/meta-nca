import functools
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import optax

from ._types import ApplyFn


@jax.jit
def masked_softmax_cross_entropy(
    logits: jax.Array, labels: jax.Array, mask: jax.Array
) -> jax.Array:
    loss = optax.softmax_cross_entropy(logits, labels)
    valid_count = jnp.sum(mask)
    # If a slice is fully padded (all mask=False), define loss as 0 for that slice.
    return jnp.sum(loss * mask.squeeze(-1)) / jnp.maximum(valid_count, 1)


@functools.partial(jax.jit, static_argnames=("apply_fn", "loss_fn"))
def compute_loss(
    batch_x: jax.Array,
    batch_y: jax.Array,
    mask: jax.Array,
    params: chex.ArrayTree,
    apply_fn: ApplyFn,
    loss_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array] = masked_softmax_cross_entropy,
) -> chex.Scalar:
    """Compute cross-entropy loss for a single tasknet."""
    logits = apply_fn(params, batch_x)
    return loss_fn(logits, batch_y, mask)
