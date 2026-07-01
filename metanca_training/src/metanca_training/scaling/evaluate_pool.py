"""Evaluate a trained local rule on a pool of architectures (per-arch val metrics)."""

from typing import Callable, Optional, Sequence

import jax
import numpy as np

import metanca
from metanca_training._validation_step import metanca_validation_step
from metanca_training.callbacks import CallbackRunner, create_accuracy_callback

from .arch_grid import arch_id, build_mlp


def evaluate_arch_pool(
    *,
    training_vars,
    local_rule_params,
    archs: Sequence[tuple[Sequence[int], str]],
    val_batches,
    cfg,
    n_update_steps: int = 10,
    n_init_samples: int = 5,
    rand_key,
    skip_ids: Optional[set[str]] = None,
    on_row: Optional[Callable[[dict], None]] = None,
) -> list[dict]:
    input_shape = tuple(cfg.dataset.input_shape)
    n_spatial_dims = len(input_shape) - 1
    n_classes = int(cfg.dataset.output_shape)
    hidden_dim = training_vars.hidden_dim
    apply_fn = training_vars.local_rule_net_apply
    shared_init = (
        training_vars.test_tasknet.hidden_state_initializer,
        training_vars.test_tasknet.hidden_dim,
    )
    skip_ids = skip_ids or set()
    use_bias = getattr(cfg.model, "use_bias", True)

    rows: list[dict] = []
    for widths, split in archs:
        if arch_id(widths) in skip_ids:
            continue
        rand_key, build_key = jax.random.split(rand_key)
        tasknet = metanca.TaskNet.build(
            model=build_mlp(widths, n_classes, use_bias=use_bias),
            input_shape=input_shape,
            key=build_key,
            n_spatial_dims=n_spatial_dims,
            d_neuron=cfg.positional_encoding.d_neuron,
            d_spatial=cfg.positional_encoding.d_spatial,
            d_layer=cfg.positional_encoding.d_layer,
            shared_initializer=shared_init,
        )
        losses, accs = [], []
        for s in range(n_init_samples):
            rand_key, eval_key = jax.random.split(rand_key)
            cb = CallbackRunner.create([create_accuracy_callback()])
            metrics, _ = metanca_validation_step(
                val_batches=val_batches,
                local_rule_net_apply=apply_fn,
                local_rule_params=local_rule_params,
                rand_key=eval_key,
                val_tasknet=tasknet,
                n_update_steps=n_update_steps,
                hidden_dim=hidden_dim,
                prop_cells_updated=cfg.training.prop_cells_updated,
                weight_transformer_dropout=cfg.training.weight_transformer_dropout,
                n_spatial_dims=n_spatial_dims,
                callback_runner=cb,
            )
            losses.append(float(metrics.get("loss", float("nan"))))
            accs.append(float(metrics.get("accuracy", float("nan"))))
        row = {
            "arch_id": arch_id(widths),
            "depth": len(widths),
            "split": split,
            "val_loss_mean": float(np.mean(losses)),
            "val_loss_std": float(np.std(losses)),
            "val_acc_mean": float(np.mean(accs)),
            "val_acc_std": float(np.std(accs)),
        }
        if on_row is not None:
            on_row(row)          # durable incremental write before continuing
        rows.append(row)
    return rows
