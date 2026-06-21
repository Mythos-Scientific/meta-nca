"""Stateful callback system for JAX-based training.

This module provides a pure functional callback system inspired by PyTorch Lightning
but designed for JAX's functional programming paradigm. All state is explicit and
immutable - callbacks are frozen dataclasses that return new instances when updated.

Example usage:

    from metanca_training.callbacks import (
        CallbackRunner,
        TrainingContext,
        create_early_stopping,
    )

    # Create callbacks
    early_stopping = create_early_stopping(
        monitor='val_loss',
        patience=10,
        mode='min',
    )

    # Create runner
    runner = CallbackRunner.create([early_stopping])

    # Initialize at training start
    ctx = TrainingContext(metaepoch=0)
    runner, result = runner.on_train_start(ctx)

    # In training loop
    for metaepoch in range(num_metaepochs):
        # ... training code ...

        # Update context with current metrics
        ctx = ctx.with_updated(
            metaepoch=metaepoch,
            metrics={'val_loss': val_loss, 'val_accuracy': val_acc},
        )

        # Run end-of-epoch callbacks
        runner, result = runner.on_epoch_end(ctx)

        if result.stop_training:
            break

    # Finalize
    runner, result = runner.on_train_end(ctx)
"""

from ._accuracy import AccuracyCallback, compute_accuracy, create_accuracy_callback
from ._base import EMPTY_RESULT, Callback, HookResult, TrainingContext, call_hook, has_hook
from ._early_stopping import EarlyStoppingCallback, create_early_stopping
from ._runner import CallbackRunner, run_callbacks
