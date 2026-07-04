"""Accuracy callback implementation.

Computes tasknet accuracies after each batch.
"""

import functools
from dataclasses import dataclass
from typing import Self

import chex
import jax
import jax.numpy as jnp

from ._base import EMPTY_RESULT, HookResult, TrainingContext


def accuracy(preds: jax.Array, targets: jax.Array, mask: jax.Array | None = None) -> chex.Scalar:
    """Compute classification accuracy (optionally masked).

    `targets` may be one-hot (same rank as `preds`, classification) or integer
    class labels (rank == preds.ndim - 1, sparse LM targets) — mirrors the
    `labels.ndim` dispatch in `_loss._auto_loss_fn`.
    """
    target_class = jnp.argmax(targets, axis=-1) if targets.ndim == preds.ndim else targets
    predicted_class = jnp.argmax(preds, axis=-1)
    correct = predicted_class == target_class
    if mask is None:
        return jnp.mean(correct)
    mask_flat = mask.squeeze(-1).astype(jnp.float32)
    valid_count = jnp.sum(mask_flat)
    return jnp.sum(correct * mask_flat) / jnp.maximum(valid_count, 1.0)


@functools.partial(jax.jit, static_argnames="apply_fn")
def compute_accuracy(
    xs: jax.Array, ys: jax.Array, mask: jax.Array | None, params: chex.ArrayTree, apply_fn
) -> chex.Scalar:
    """Compute accuracy for a single tasknet."""
    logits = apply_fn(params, xs)
    return accuracy(logits, ys, mask=mask)


@dataclass(frozen=True)
class AccuracyCallback:
    """Callback that computes tasknet accuracies after each batch.

    This callback computes the mean accuracy across all tasknets and
    returns it in the logs. It requires the TrainingContext to have
    tasknet_data_list, apply_fns, batch_x, and batch_y populated.

    This is a stateless callback - it doesn't track any state across
    batches, just computes metrics.
    """

    def on_batch_end(self, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Compute accuracies after each batch."""
        if ctx.tasknet_data_list is None or ctx.apply_fns is None:
            return self, EMPTY_RESULT

        if ctx.batch_x is None or ctx.batch_y is None:
            return self, EMPTY_RESULT

        accuracies = jnp.zeros(len(ctx.tasknet_data_list))
        for i, ((params, _, _), apply_fn) in enumerate(zip(ctx.tasknet_data_list, ctx.apply_fns)):
            acc = compute_accuracy(ctx.batch_x, ctx.batch_y, ctx.mask, params, apply_fn)
            accuracies = accuracies.at[i].set(acc)

        mean_accuracy = accuracies.mean()

        return self, HookResult(
            logs={"accuracy": mean_accuracy},
            data={"accuracies": accuracies},
        )


def create_accuracy_callback() -> AccuracyCallback:
    """Factory function to create an AccuracyCallback."""
    return AccuracyCallback()
