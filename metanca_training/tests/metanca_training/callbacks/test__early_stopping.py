"""Tests for the EarlyStoppingCallback."""

import math

from metanca_training.callbacks import CallbackRunner, TrainingContext, create_early_stopping


class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""

    def test_create_early_stopping_defaults(self):
        """Test factory function with defaults."""
        callback = create_early_stopping()
        assert callback.monitor == "val_loss"
        assert callback.patience == 10
        assert callback.min_delta == 0.0
        assert callback.mode == "min"
        assert callback.baseline is None

    def test_create_early_stopping_custom(self):
        """Test factory function with custom values."""
        callback = create_early_stopping(
            monitor="val_accuracy",
            patience=5,
            min_delta=0.01,
            mode="max",
            baseline=0.8,
        )
        assert callback.monitor == "val_accuracy"
        assert callback.patience == 5
        assert callback.min_delta == 0.01
        assert callback.mode == "max"
        assert callback.baseline == 0.8

    def test_on_train_start_initializes_state(self):
        """Test that on_train_start properly initializes state."""
        callback = create_early_stopping(mode="min")
        ctx = TrainingContext()

        new_callback, result = callback.on_train_start(ctx)

        assert new_callback._initialized
        assert new_callback.best_value == math.inf
        assert new_callback.wait_count == 0
        assert not result.stop_training

    def test_on_train_start_max_mode(self):
        """Test initialization in max mode."""
        callback = create_early_stopping(mode="max")
        ctx = TrainingContext()

        new_callback, _ = callback.on_train_start(ctx)

        assert new_callback.best_value == -math.inf

    def test_on_train_start_with_baseline(self):
        """Test initialization with baseline."""
        callback = create_early_stopping(baseline=0.5)
        ctx = TrainingContext()

        new_callback, _ = callback.on_train_start(ctx)

        assert new_callback.best_value == 0.5

    def test_improvement_resets_wait_count(self):
        """Test that improvement resets the wait counter."""
        callback = create_early_stopping(monitor="val_loss", patience=3, mode="min")
        ctx = TrainingContext()

        callback, _ = callback.on_train_start(ctx)

        # First epoch - metric improves
        ctx = ctx.with_updated(metaepoch=1, metrics={"val_loss": 0.5})
        callback, result = callback.on_epoch_end(ctx)
        assert callback.best_value == 0.5
        assert callback.wait_count == 0
        assert not result.stop_training

        # Second epoch - no improvement
        ctx = ctx.with_updated(metaepoch=2, metrics={"val_loss": 0.6})
        callback, result = callback.on_epoch_end(ctx)
        assert callback.wait_count == 1
        assert not result.stop_training

        # Third epoch - improvement!
        ctx = ctx.with_updated(metaepoch=3, metrics={"val_loss": 0.4})
        callback, result = callback.on_epoch_end(ctx)
        assert callback.best_value == 0.4
        assert callback.wait_count == 0
        assert not result.stop_training

    def test_stops_after_patience_exceeded(self):
        """Test that training stops after patience is exceeded."""
        callback = create_early_stopping(monitor="val_loss", patience=3, mode="min")
        ctx = TrainingContext()

        callback, _ = callback.on_train_start(ctx)

        # First epoch - improvement
        ctx = ctx.with_updated(metaepoch=0, metrics={"val_loss": 0.5})
        callback, _ = callback.on_epoch_end(ctx)

        # Next 3 epochs - no improvement (exceeds patience)
        for i in range(1, 4):
            ctx = ctx.with_updated(metaepoch=i, metrics={"val_loss": 0.6})
            callback, result = callback.on_epoch_end(ctx)

            if i < 3:
                assert not result.stop_training
            else:
                assert result.stop_training
                assert callback.stopped_epoch == 3

    def test_max_mode_improvement_detection(self):
        """Test improvement detection in max mode."""
        callback = create_early_stopping(monitor="val_accuracy", patience=3, mode="max")
        ctx = TrainingContext()

        callback, _ = callback.on_train_start(ctx)

        # Improvement (higher is better)
        ctx = ctx.with_updated(metaepoch=0, metrics={"val_accuracy": 0.7})
        callback, _ = callback.on_epoch_end(ctx)
        assert callback.best_value == 0.7

        # Another improvement
        ctx = ctx.with_updated(metaepoch=1, metrics={"val_accuracy": 0.8})
        callback, _ = callback.on_epoch_end(ctx)
        assert callback.best_value == 0.8

        # No improvement
        ctx = ctx.with_updated(metaepoch=2, metrics={"val_accuracy": 0.75})
        callback, _ = callback.on_epoch_end(ctx)
        assert callback.best_value == 0.8
        assert callback.wait_count == 1

    def test_min_delta_threshold(self):
        """Test that min_delta is respected."""
        callback = create_early_stopping(monitor="val_loss", patience=3, mode="min", min_delta=0.1)
        ctx = TrainingContext()

        callback, _ = callback.on_train_start(ctx)

        # Initial value
        ctx = ctx.with_updated(metaepoch=0, metrics={"val_loss": 1.0})
        callback, _ = callback.on_epoch_end(ctx)
        assert callback.best_value == 1.0

        # Small improvement - not enough to beat min_delta
        ctx = ctx.with_updated(metaepoch=1, metrics={"val_loss": 0.95})
        callback, _ = callback.on_epoch_end(ctx)
        assert callback.best_value == 1.0  # Not updated
        assert callback.wait_count == 1

        # Large improvement - beats min_delta
        ctx = ctx.with_updated(metaepoch=2, metrics={"val_loss": 0.85})
        callback, _ = callback.on_epoch_end(ctx)
        assert callback.best_value == 0.85
        assert callback.wait_count == 0

    def test_missing_metric_skips_check(self):
        """Test that missing metric doesn't cause issues."""
        callback = create_early_stopping(monitor="val_loss")
        ctx = TrainingContext()

        callback, _ = callback.on_train_start(ctx)

        # Metric not present
        ctx = ctx.with_updated(metaepoch=0, metrics={"train_loss": 0.5})
        callback, result = callback.on_epoch_end(ctx)

        assert callback.wait_count == 0
        assert not result.stop_training

    def test_immutability(self):
        """Test that callback state is immutable."""
        original = create_early_stopping()
        ctx = TrainingContext()

        updated, _ = original.on_train_start(ctx)

        # Original should be unchanged
        assert not original._initialized
        assert updated._initialized

    def test_best_epoch_tracking(self):
        """Test that best_epoch is correctly tracked."""
        callback = create_early_stopping(monitor="val_loss", mode="min")
        ctx = TrainingContext()

        callback, _ = callback.on_train_start(ctx)

        # Epoch 0 - first value
        ctx = ctx.with_updated(metaepoch=0, metrics={"val_loss": 0.5})
        callback, _ = callback.on_epoch_end(ctx)
        assert callback.best_epoch == 0

        # Epoch 5 - improvement
        ctx = ctx.with_updated(metaepoch=5, metrics={"val_loss": 0.3})
        callback, _ = callback.on_epoch_end(ctx)
        assert callback.best_epoch == 5

        # Epoch 10 - no improvement
        ctx = ctx.with_updated(metaepoch=10, metrics={"val_loss": 0.4})
        callback, _ = callback.on_epoch_end(ctx)
        assert callback.best_epoch == 5  # Still epoch 5


class TestCallbackRunner:
    """Tests for CallbackRunner."""

    def test_create_runner(self):
        """Test creating a runner with callbacks."""
        cb1 = create_early_stopping(monitor="val_loss")
        cb2 = create_early_stopping(monitor="val_accuracy", mode="max")

        runner = CallbackRunner.create([cb1, cb2])

        assert len(runner) == 2

    def test_runner_executes_all_callbacks(self):
        """Test that runner executes hooks on all callbacks."""
        cb1 = create_early_stopping(monitor="val_loss", patience=2)
        cb2 = create_early_stopping(monitor="val_accuracy", mode="max", patience=2)

        runner = CallbackRunner.create([cb1, cb2])
        ctx = TrainingContext()

        runner, _ = runner.on_train_start(ctx)

        # Both should be initialized
        for callback in runner.callbacks:
            assert callback._initialized

    def test_runner_merges_results(self):
        """Test that runner properly merges HookResults."""
        cb1 = create_early_stopping(monitor="val_loss", patience=1)
        cb2 = create_early_stopping(monitor="val_accuracy", mode="max", patience=1)

        runner = CallbackRunner.create([cb1, cb2])
        ctx = TrainingContext()

        runner, _ = runner.on_train_start(ctx)

        # First epoch
        ctx = ctx.with_updated(metaepoch=0, metrics={"val_loss": 0.5, "val_accuracy": 0.8})
        runner, _ = runner.on_epoch_end(ctx)

        # Second epoch - no improvement for either
        ctx = ctx.with_updated(metaepoch=1, metrics={"val_loss": 0.6, "val_accuracy": 0.7})
        runner, result = runner.on_epoch_end(ctx)

        # Both should trigger stop
        assert result.stop_training

    def test_runner_stop_training_propagates(self):
        """Test that stop_training from any callback propagates."""
        cb1 = create_early_stopping(monitor="val_loss", patience=1)
        cb2 = create_early_stopping(monitor="val_accuracy", mode="max", patience=100)

        runner = CallbackRunner.create([cb1, cb2])
        ctx = TrainingContext()

        runner, _ = runner.on_train_start(ctx)

        # First epoch
        ctx = ctx.with_updated(metaepoch=0, metrics={"val_loss": 0.5, "val_accuracy": 0.8})
        runner, _ = runner.on_epoch_end(ctx)

        # Second epoch - only val_loss triggers
        ctx = ctx.with_updated(metaepoch=1, metrics={"val_loss": 0.6, "val_accuracy": 0.9})
        runner, result = runner.on_epoch_end(ctx)

        # Should stop because cb1 triggered
        assert result.stop_training

    def test_empty_runner(self):
        """Test runner with no callbacks."""
        runner = CallbackRunner.create([])
        ctx = TrainingContext()

        runner, result = runner.on_train_start(ctx)

        assert len(runner) == 0
        assert not result.stop_training


class TestTrainingContext:
    """Tests for TrainingContext."""

    def test_with_updated(self):
        """Test with_updated creates new context."""
        ctx = TrainingContext(metaepoch=0, metrics={"loss": 1.0})

        new_ctx = ctx.with_updated(metaepoch=1, metrics={"loss": 0.5})

        assert ctx.metaepoch == 0
        assert ctx.metrics["loss"] == 1.0
        assert new_ctx.metaepoch == 1
        assert new_ctx.metrics["loss"] == 0.5

    def test_default_values(self):
        """Test default context values."""
        ctx = TrainingContext()

        assert ctx.metaepoch == 0
        assert ctx.batch_idx == 0
        assert ctx.n_batches == 0
        assert ctx.metrics == {}
        assert ctx.local_rule_params is None
        assert ctx.optimizer_state is None
