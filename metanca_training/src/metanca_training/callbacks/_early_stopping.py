"""Early stopping callback implementation.

Monitors a metric and stops training when it stops improving.
"""

import math
from dataclasses import dataclass
from typing import Literal, Self

from ._base import EMPTY_RESULT, HookResult, TrainingContext


@dataclass(frozen=True)
class EarlyStoppingCallback:
    """Early stopping callback that monitors a metric and stops training.

    This is a frozen dataclass - all state updates return new instances.
    The callback checks the monitored metric at the end of each epoch
    (or after validation) and tracks improvement.

    Attributes:
        monitor: Name of the metric to monitor (e.g., 'val_loss', 'val_accuracy').
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' to minimize the metric, 'max' to maximize it.
        baseline: Optional baseline value. Training stops if metric doesn't
            improve beyond this.
        restore_best: If True, logs a message about best weights (actual
            restoration would be handled by checkpoint callback).

        # State fields (managed internally)
        best_value: Best metric value seen so far.
        wait_count: Number of epochs since last improvement.
        stopped_epoch: Epoch at which training was stopped (0 if not stopped).
        best_epoch: Epoch at which best value was observed.
    """

    # Configuration (set at creation, immutable)
    monitor: str = "val_loss"
    patience: int = 10
    min_delta: float = 0.0
    mode: Literal["min", "max"] = "min"
    baseline: float | None = None
    restore_best: bool = True

    # State (updated via with_updated pattern)
    best_value: float = math.inf  # Will be set properly on first call
    wait_count: int = 0
    stopped_epoch: int = 0
    best_epoch: int = 0
    _initialized: bool = False

    def with_updated(self, **kwargs) -> Self:
        """Return a new instance with updated fields."""
        return EarlyStoppingCallback(
            monitor=kwargs.get("monitor", self.monitor),
            patience=kwargs.get("patience", self.patience),
            min_delta=kwargs.get("min_delta", self.min_delta),
            mode=kwargs.get("mode", self.mode),
            baseline=kwargs.get("baseline", self.baseline),
            restore_best=kwargs.get("restore_best", self.restore_best),
            best_value=kwargs.get("best_value", self.best_value),
            wait_count=kwargs.get("wait_count", self.wait_count),
            stopped_epoch=kwargs.get("stopped_epoch", self.stopped_epoch),
            best_epoch=kwargs.get("best_epoch", self.best_epoch),
            _initialized=kwargs.get("_initialized", self._initialized),
        )

    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement over best."""
        if self.mode == "min":
            return current < (self.best_value - self.min_delta)
        else:  # mode == "max"
            return current > (self.best_value + self.min_delta)

    def _get_initial_best(self) -> float:
        """Get the initial best value based on mode and baseline."""
        if self.baseline is not None:
            return self.baseline
        return math.inf if self.mode == "min" else -math.inf

    def on_train_start(self, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Initialize state at the start of training."""
        new_self = self.with_updated(
            best_value=self._get_initial_best(),
            wait_count=0,
            stopped_epoch=0,
            best_epoch=0,
            _initialized=True,
        )
        return new_self, EMPTY_RESULT

    def on_epoch_end(self, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Check for improvement at end of epoch."""
        return self._check_metric(ctx)

    def on_validation_end(self, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Check for improvement after validation (alternative to on_epoch_end)."""
        # Only check here if the monitored metric starts with 'val_'
        # This avoids double-checking if both hooks are called
        if self.monitor.startswith("val_"):
            return self._check_metric(ctx)
        return self, EMPTY_RESULT

    def _check_metric(self, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Core logic for checking metric and updating state."""
        # Initialize if needed (in case on_train_start wasn't called)
        if not self._initialized:
            new_self = self.with_updated(
                best_value=self._get_initial_best(),
                _initialized=True,
            )
            return new_self._check_metric(ctx)

        # Get current metric value
        current_value = ctx.metrics.get(self.monitor)
        if current_value is None:
            # Metric not available yet, skip check
            return self, EMPTY_RESULT

        logs: dict = {}

        # Check for improvement
        if self._is_improvement(current_value):
            # Improvement found
            new_self = self.with_updated(
                best_value=current_value,
                best_epoch=ctx.metaepoch,
                wait_count=0,
            )
            logs["early_stopping_best_value"] = current_value
            logs["early_stopping_best_epoch"] = ctx.metaepoch
            return new_self, HookResult(logs=logs)

        # No improvement
        new_wait = self.wait_count + 1

        if new_wait >= self.patience:
            # Patience exceeded, stop training
            new_self = self.with_updated(
                wait_count=new_wait,
                stopped_epoch=ctx.metaepoch,
            )
            logs["early_stopping_stopped_epoch"] = ctx.metaepoch
            logs["early_stopping_best_value"] = self.best_value
            logs["early_stopping_best_epoch"] = self.best_epoch

            return new_self, HookResult(
                stop_training=True,
                logs=logs,
                data={
                    "early_stopping_message": (
                        f"Early stopping triggered at epoch {ctx.metaepoch}. "
                        f"Best {self.monitor}={self.best_value:.6f} at epoch {self.best_epoch}."
                    )
                },
            )

        # Still waiting
        new_self = self.with_updated(wait_count=new_wait)
        logs["early_stopping_wait_count"] = new_wait
        return new_self, HookResult(logs=logs)

    def on_train_end(self, ctx: TrainingContext) -> tuple[Self, HookResult]:
        """Report final state at end of training."""
        if self.stopped_epoch > 0:
            msg = (
                f"Training stopped early at epoch {self.stopped_epoch}. "
                f"Best {self.monitor}={self.best_value:.6f} at epoch {self.best_epoch}."
            )
        else:
            msg = (
                f"Training completed. "
                f"Best {self.monitor}={self.best_value:.6f} at epoch {self.best_epoch}."
            )

        return self, HookResult(
            logs={
                "early_stopping_final_best_value": self.best_value,
                "early_stopping_final_best_epoch": self.best_epoch,
            },
            data={"early_stopping_summary": msg},
        )


def create_early_stopping(
    monitor: str = "val_loss",
    patience: int = 10,
    min_delta: float = 0.0,
    mode: Literal["min", "max"] = "min",
    baseline: float | None = None,
    restore_best: bool = True,
) -> EarlyStoppingCallback:
    """Factory function to create an EarlyStoppingCallback.

    Args:
        monitor: Metric name to monitor.
        patience: Epochs to wait for improvement before stopping.
        min_delta: Minimum change to count as improvement.
        mode: 'min' for metrics to minimize, 'max' to maximize.
        baseline: Optional baseline - stop if metric doesn't beat this.
        restore_best: Whether to recommend restoring best weights.

    Returns:
        Configured EarlyStoppingCallback instance.

    Example:
        >>> callback = create_early_stopping(
        ...     monitor='val_accuracy',
        ...     patience=5,
        ...     mode='max',
        ... )
    """
    return EarlyStoppingCallback(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        mode=mode,
        baseline=baseline,
        restore_best=restore_best,
    )
