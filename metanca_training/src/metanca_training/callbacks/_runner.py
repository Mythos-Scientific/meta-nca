"""Callback runner for orchestrating multiple callbacks.

The CallbackRunner manages a collection of callbacks, executing their hooks
in sequence and aggregating results. All state is explicit and immutable.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Self, Sequence

from ._base import EMPTY_RESULT, HookResult, TrainingContext, call_hook

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CallbackRunner:
    """Orchestrates execution of multiple callbacks.

    This is a frozen dataclass that holds callback states. When hooks are
    executed, a new CallbackRunner is returned with updated callback states.

    Attributes:
        callbacks: Tuple of callback instances (immutable sequence).

    Example:
        >>> runner = CallbackRunner.create([
        ...     create_early_stopping(monitor='val_loss', patience=5),
        ...     create_checkpoint_callback(save_dir='./checkpoints'),
        ... ])
        >>>
        >>> # At start of training
        >>> runner, result = runner.on_train_start(ctx)
        >>>
        >>> # After each epoch
        >>> runner, result = runner.on_epoch_end(ctx)
        >>> if result.stop_training:
        ...     break
    """

    callbacks: tuple[Any, ...] = field(default_factory=tuple)

    @classmethod
    def create(cls, callbacks: Sequence[Any]) -> "CallbackRunner":
        """Create a CallbackRunner from a sequence of callbacks."""
        return cls(callbacks=tuple(callbacks))

    def with_updated_callbacks(self, callbacks: Sequence[Any]) -> Self:
        """Return a new runner with updated callbacks."""
        return CallbackRunner(callbacks=tuple(callbacks))

    def _run_hook(self, hook_name: str, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Execute a hook on all callbacks and aggregate results.

        Args:
            hook_name: Name of the hook method to call.
            ctx: Training context to pass to hooks.

        Returns:
            Tuple of (new_runner_with_updated_states, merged_hook_result).
        """
        updated_callbacks = []
        merged_result = EMPTY_RESULT

        for callback in self.callbacks:
            new_callback, result = call_hook(callback, hook_name, ctx)
            updated_callbacks.append(new_callback)
            merged_result = merged_result.merge(result)

            # Log any messages from callbacks
            if "early_stopping_message" in result.data:
                logger.info(result.data["early_stopping_message"])
            if "early_stopping_summary" in result.data:
                logger.info(result.data["early_stopping_summary"])

        new_runner = self.with_updated_callbacks(updated_callbacks)
        return new_runner, merged_result

    def on_train_start(self, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Execute on_train_start hook on all callbacks."""
        return self._run_hook("on_train_start", ctx)

    def on_train_end(self, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Execute on_train_end hook on all callbacks."""
        return self._run_hook("on_train_end", ctx)

    def on_epoch_start(self, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Execute on_epoch_start hook on all callbacks."""
        return self._run_hook("on_epoch_start", ctx)

    def on_epoch_end(self, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Execute on_epoch_end hook on all callbacks."""
        return self._run_hook("on_epoch_end", ctx)

    def on_batch_start(self, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Execute on_batch_start hook on all callbacks."""
        return self._run_hook("on_batch_start", ctx)

    def on_batch_end(self, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Execute on_batch_end hook on all callbacks."""
        return self._run_hook("on_batch_end", ctx)

    def on_validation_end(self, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Execute on_validation_end hook on all callbacks."""
        return self._run_hook("on_validation_end", ctx)

    def __len__(self) -> int:
        """Return number of callbacks."""
        return len(self.callbacks)

    def __iter__(self):
        """Iterate over callbacks."""
        return iter(self.callbacks)


def run_callbacks(
    callbacks: Sequence[Any],
    hook_name: str,
    ctx: TrainingContext,
) -> tuple[list[Any], HookResult]:
    """Functional interface for running callbacks without a runner.

    This is a convenience function for cases where you don't want to
    maintain a CallbackRunner instance.

    Args:
        callbacks: Sequence of callback instances.
        hook_name: Name of the hook to execute.
        ctx: Training context.

    Returns:
        Tuple of (list of updated callbacks, merged hook result).

    Example:
        >>> callbacks = [early_stopping, checkpoint]
        >>> callbacks, result = run_callbacks(callbacks, 'on_epoch_end', ctx)
        >>> if result.stop_training:
        ...     break
    """
    runner = CallbackRunner.create(callbacks)
    new_runner, result = runner._run_hook(hook_name, ctx)
    return list(new_runner.callbacks), result
