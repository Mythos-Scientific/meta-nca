"""Base types for the stateful callback system.

This module provides a pure functional callback system for JAX-based training.
All state is explicit and immutable - callbacks are frozen dataclasses that
return new instances when their state changes.

Design principles:
1. Callbacks are frozen dataclasses containing both config and state
2. Hook methods are pure functions: (self, context) -> (new_self, HookResult)
3. HookResult carries signals (stop_training) and data (logs)
4. TrainingContext provides read-only access to training state
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, Self, Sequence, runtime_checkable

import chex
import jax


@dataclass(frozen=True)
class HookResult:
    """Result returned by callback hooks.

    Attributes:
        stop_training: If True, signals the training loop to stop.
        logs: Dictionary of values to log (e.g., to wandb).
        data: Arbitrary data that can be passed to subsequent hooks.
    """

    stop_training: bool = False
    logs: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "HookResult") -> "HookResult":
        """Merge two HookResults, combining logs and data."""
        return HookResult(
            stop_training=self.stop_training or other.stop_training,
            logs={**self.logs, **other.logs},
            data={**self.data, **other.data},
        )


@dataclass(frozen=True)
class TrainingContext:
    """Read-only context passed to callback hooks.

    This provides callbacks with access to training state without
    allowing direct modification. Callbacks signal desired changes
    through HookResult.

    Attributes:
        metaepoch: Current metaepoch index.
        batch_idx: Current batch index within the metaepoch.
        n_batches: Total number of batches per metaepoch.
        metrics: Current metrics dictionary.
        local_rule_params: Current local rule network parameters.
        optimizer_state: Current optimizer state.
        tasknet_data_list: List of (params, hidden_states, positional_encodings) tuples.
        apply_fns: Sequence of tasknet apply functions.
        batch_x: Current batch input data (shared across all tasknets).
        batch_y: Current batch target data (shared across all tasknets).
        mask: Current batch mask (shared across all tasknets).
        per_arch_batches: Optional per-tasknet batches, aligned 1:1 with
            `tasknet_data_list`/`apply_fns`, as a sequence of (x, y, mask)
            tuples. Used by dict-mode (mixed-vocab LLM pools) where each
            arch trains on its own token stream and must be scored on that
            SAME stream rather than on a single shared representative batch.
            When None (the default — tuple-mode and validation), callbacks
            fall back to the shared `batch_x`/`batch_y`/`mask` fields with
            behavior identical to before this field existed.
    """

    metaepoch: int = 0
    batch_idx: int = 0
    n_batches: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    local_rule_params: chex.ArrayTree | None = None
    optimizer_state: Any | None = None
    tasknet_data_list: list[tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree]] | None = None
    apply_fns: Sequence[Any] | None = None
    batch_x: jax.Array | None = None
    batch_y: jax.Array | None = None
    mask: jax.Array | None = None
    per_arch_batches: Sequence[tuple[jax.Array, jax.Array, jax.Array]] | None = None

    def with_updated(self, **kwargs) -> "TrainingContext":
        """Return a new TrainingContext with updated fields."""
        return TrainingContext(
            metaepoch=kwargs.get("metaepoch", self.metaepoch),
            batch_idx=kwargs.get("batch_idx", self.batch_idx),
            n_batches=kwargs.get("n_batches", self.n_batches),
            metrics=kwargs.get("metrics", self.metrics),
            local_rule_params=kwargs.get("local_rule_params", self.local_rule_params),
            optimizer_state=kwargs.get("optimizer_state", self.optimizer_state),
            tasknet_data_list=kwargs.get("tasknet_data_list", self.tasknet_data_list),
            apply_fns=kwargs.get("apply_fns", self.apply_fns),
            batch_x=kwargs.get("batch_x", self.batch_x),
            batch_y=kwargs.get("batch_y", self.batch_y),
            mask=kwargs.get("mask", self.mask),
            per_arch_batches=kwargs.get("per_arch_batches", self.per_arch_batches),
        )


# Empty result singleton for hooks that don't need to return anything
EMPTY_RESULT = HookResult()


@runtime_checkable
class Callback(Protocol):
    """Protocol defining the callback interface.

    Callbacks should be frozen dataclasses implementing any subset of these hooks.
    Each hook takes the current callback state and training context,
    returning a tuple of (new_callback_state, HookResult).

    Hooks are optional - if not implemented, they are skipped.
    """

    def on_train_start(self, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Called once before training begins."""
        ...

    def on_train_end(self, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Called once after training ends."""
        ...

    def on_epoch_start(self, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Called at the start of each metaepoch."""
        ...

    def on_epoch_end(self, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Called at the end of each metaepoch."""
        ...

    def on_batch_start(self, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Called before each batch."""
        ...

    def on_batch_end(self, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Called after each batch."""
        ...

    def on_validation_end(self, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Called after validation is complete."""
        ...


def has_hook(callback: Any, hook_name: str) -> bool:
    """Check if a callback implements a specific hook.

    Returns True if the callback has the hook method and it's not
    just the default Protocol stub.
    """
    if not hasattr(callback, hook_name):
        return False

    method = getattr(callback, hook_name)
    # Check if it's a real implementation (not just inherited from Protocol)
    if hasattr(method, "__func__"):
        # Bound method - check if defined on the actual class
        return hook_name in type(callback).__dict__
    return callable(method)


def call_hook(callback: Any, hook_name: str, ctx: TrainingContext) -> tuple[Any, HookResult]:
    """Safely call a hook on a callback.

    If the hook doesn't exist, returns (callback, EMPTY_RESULT).
    """
    if not has_hook(callback, hook_name):
        return callback, EMPTY_RESULT

    method = getattr(callback, hook_name)
    return method(ctx)
