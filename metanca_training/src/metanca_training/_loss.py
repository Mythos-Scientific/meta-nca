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


@jax.jit
def masked_sparse_softmax_cross_entropy(
    logits: jax.Array, labels: jax.Array, mask: jax.Array
) -> jax.Array:
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    valid_count = jnp.sum(mask)
    return jnp.sum(loss * mask.squeeze(-1)) / jnp.maximum(valid_count, 1)


def _auto_loss_fn(
    logits: jax.Array,
    labels: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    if labels.ndim == logits.ndim:
        return masked_softmax_cross_entropy(logits, labels, mask)
    if labels.ndim == logits.ndim - 1:
        return masked_sparse_softmax_cross_entropy(logits, labels, mask)
    raise ValueError(
        f"Expected labels rank {logits.ndim} (one-hot) or {logits.ndim - 1} "
        f"(integer labels), got {labels.ndim}"
    )


@functools.partial(jax.jit, static_argnames=("apply_fn", "loss_fn"))
def compute_loss(
    batch_x: jax.Array,
    batch_y: jax.Array,
    mask: jax.Array,
    params: chex.ArrayTree,
    apply_fn: ApplyFn,
    loss_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array] | None = None,
) -> chex.Scalar:
    """Compute cross-entropy loss for a single tasknet."""
    logits = apply_fn(params, batch_x)
    resolved_loss_fn = _auto_loss_fn if loss_fn is None else loss_fn
    return resolved_loss_fn(logits, batch_y, mask)
