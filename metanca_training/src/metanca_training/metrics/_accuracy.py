"""Accuracy metric."""

import chex
import jax
import jax.numpy as jnp


def accuracy(logits: jax.Array, targets: jax.Array, mask: jax.Array | None = None) -> chex.Scalar:
    """Compute classification accuracy from logits and labels."""
    if targets.ndim == logits.ndim:
        target_class = jnp.argmax(targets, axis=-1)
    elif targets.ndim == logits.ndim - 1:
        target_class = targets
    else:
        raise ValueError(
            f"Expected targets rank {logits.ndim} (one-hot) or {logits.ndim - 1} "
            f"(integer labels), got {targets.ndim}"
        )

    predicted_class = jnp.argmax(logits, axis=-1)
    correct = predicted_class == target_class

    if mask is None:
        return jnp.mean(correct)

    mask_flat = mask.squeeze(-1).astype(jnp.float32)
    valid_count = jnp.sum(mask_flat)
    return jnp.sum(correct * mask_flat) / jnp.maximum(valid_count, 1.0)


def compute_accuracy(
    *,
    logits: jax.Array,
    labels: jax.Array,
    mask: jax.Array | None = None,
    **_: object,
) -> dict[str, chex.Scalar]:
    """Return accuracy in the standard metric-dictionary form."""
    return {"accuracy": accuracy(logits, labels, mask)}
