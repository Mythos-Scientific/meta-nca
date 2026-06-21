"""Tests for the AccuracyCallback."""

import jax.numpy as jnp

from metanca_training.callbacks import AccuracyCallback, TrainingContext, create_accuracy_callback


class TestAccuracyCallback:
    """Tests for AccuracyCallback."""

    def test_create_accuracy_callback(self):
        """Test factory function."""
        callback = create_accuracy_callback()
        assert isinstance(callback, AccuracyCallback)

    def test_on_batch_end_missing_data(self):
        """Test that callback handles missing data gracefully."""
        callback = create_accuracy_callback()
        ctx = TrainingContext()

        new_callback, result = callback.on_batch_end(ctx)

        assert new_callback is callback
        assert not result.stop_training
        assert result.logs == {}

    def test_on_batch_end_missing_apply_fns(self):
        """Test that callback handles missing apply_fns."""
        callback = create_accuracy_callback()
        ctx = TrainingContext(
            tasknet_data_list=[({}, {}, {})],
            batch_x=jnp.zeros((2, 10)),
            batch_y=jnp.zeros((2, 3)),
        )

        new_callback, result = callback.on_batch_end(ctx)

        assert result.logs == {}

    def test_on_batch_end_missing_batch_data(self):
        """Test that callback handles missing batch data."""
        callback = create_accuracy_callback()

        def dummy_apply(params, x):
            return x

        ctx = TrainingContext(
            tasknet_data_list=[({}, {}, {})],
            apply_fns=[dummy_apply],
        )

        new_callback, result = callback.on_batch_end(ctx)

        assert result.logs == {}

    def test_on_batch_end_computes_accuracy(self):
        """Test that callback computes accuracy correctly."""
        callback = create_accuracy_callback()

        # Create a simple apply function that returns logits
        def dummy_apply(params, x):
            # Return logits where class 0 has highest score
            batch_size = x.shape[0]
            return jnp.array([[2.0, 1.0, 0.0]] * batch_size)

        # Targets where class 0 is correct
        batch_y = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        ctx = TrainingContext(
            tasknet_data_list=[({}, {}, {})],
            apply_fns=[dummy_apply],
            batch_x=jnp.zeros((2, 10)),
            batch_y=batch_y,
        )

        new_callback, result = callback.on_batch_end(ctx)

        assert "accuracy" in result.logs
        assert result.logs["accuracy"] == 1.0  # 100% accuracy

    def test_on_batch_end_partial_accuracy(self):
        """Test accuracy computation with partial correctness."""
        callback = create_accuracy_callback()

        def dummy_apply(params, x):
            # First sample predicts class 0, second predicts class 1
            return jnp.array([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])

        # First target is class 0 (correct), second is class 0 (incorrect)
        batch_y = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        ctx = TrainingContext(
            tasknet_data_list=[({}, {}, {})],
            apply_fns=[dummy_apply],
            batch_x=jnp.zeros((2, 10)),
            batch_y=batch_y,
        )

        new_callback, result = callback.on_batch_end(ctx)

        assert "accuracy" in result.logs
        assert result.logs["accuracy"] == 0.5  # 50% accuracy

    def test_on_batch_end_multiple_tasknets(self):
        """Test accuracy averaging across multiple tasknets."""
        callback = create_accuracy_callback()

        def apply_100_pct(params, x):
            return jnp.array([[2.0, 0.0]])  # Predicts class 0

        def apply_0_pct(params, x):
            return jnp.array([[0.0, 2.0]])  # Predicts class 1

        batch_y = jnp.array([[1.0, 0.0]])  # Target is class 0

        ctx = TrainingContext(
            tasknet_data_list=[({}, {}, {}), ({}, {}, {})],
            apply_fns=[apply_100_pct, apply_0_pct],
            batch_x=jnp.zeros((1, 10)),
            batch_y=batch_y,
        )

        new_callback, result = callback.on_batch_end(ctx)

        assert "accuracy" in result.logs
        # Average of 100% and 0% = 50%
        assert result.logs["accuracy"] == 0.5

    def test_stateless_callback(self):
        """Test that callback is stateless (same instance returned)."""
        callback = create_accuracy_callback()

        def dummy_apply(params, x):
            return jnp.array([[1.0, 0.0]])

        ctx = TrainingContext(
            tasknet_data_list=[({}, {}, {})],
            apply_fns=[dummy_apply],
            batch_x=jnp.zeros((1, 10)),
            batch_y=jnp.array([[1.0, 0.0]]),
        )

        new_callback, _ = callback.on_batch_end(ctx)

        # Should return same instance since it's stateless
        assert new_callback is callback
