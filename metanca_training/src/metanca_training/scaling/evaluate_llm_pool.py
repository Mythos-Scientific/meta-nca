"""Evaluate a trained local rule on a pool of LLM architectures (per-arch val metrics)."""

import math
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

import metanca
from metanca_training._validation_step import metanca_validation_step
from metanca_training.callbacks import CallbackRunner

from .llm_grid import LLMArch, build_tiny_lm, llm_arch_id


def evaluate_llm_pool(
    *,
    training_vars,
    local_rule_params,
    archs: list[tuple[LLMArch, str]],
    val_batches: dict[int, tuple],
    factors: dict,
    cfg,
    n_update_steps: int = 10,
    n_init_samples: int = 5,
    rand_key,
    skip_ids: Optional[set[str]] = None,
    on_row: Optional[Callable[[dict], None]] = None,
) -> list[dict]:
    context_length = int(cfg.dataset.input_shape[0])
    hidden_dim = training_vars.hidden_dim
    apply_fn = training_vars.local_rule_net_apply
    shared_init = (
        training_vars.test_tasknet.hidden_state_initializer,
        training_vars.test_tasknet.hidden_dim,
    )
    skip_ids = skip_ids or set()

    rows: list[dict] = []
    for arch, split in archs:
        aid = llm_arch_id(arch)
        if aid in skip_ids:
            continue
        rand_key, build_key = jax.random.split(rand_key)
        tasknet = metanca.TaskNet.build(
            model=build_tiny_lm(arch, context_length),
            input_shape=(context_length,),
            key=build_key,
            n_spatial_dims=0,
            d_neuron=cfg.positional_encoding.d_neuron,
            d_spatial=cfg.positional_encoding.d_spatial,
            d_layer=cfg.positional_encoding.d_layer,
            shared_initializer=shared_init,
            dummy_input_dtype=jnp.int32,
        )
        arch_val_batches = val_batches[arch.vocab]
        losses, ppls, bpbs = [], [], []
        for _ in range(n_init_samples):
            rand_key, eval_key = jax.random.split(rand_key)
            cb = CallbackRunner.create([])
            metrics, _ = metanca_validation_step(
                val_batches=arch_val_batches,
                local_rule_net_apply=apply_fn,
                local_rule_params=local_rule_params,
                rand_key=eval_key,
                val_tasknet=tasknet,
                n_update_steps=n_update_steps,
                hidden_dim=hidden_dim,
                prop_cells_updated=cfg.training.prop_cells_updated,
                weight_transformer_dropout=cfg.training.weight_transformer_dropout,
                n_spatial_dims=0,
                callback_runner=cb,
            )
            loss = float(metrics.get("loss", float("nan")))
            losses.append(loss)
            ppls.append(math.exp(loss))
            bpbs.append(loss * factors[arch.vocab]["val"] / math.log(2))
        row = {
            "arch_id": aid,
            "d_model": arch.d_model,
            "num_heads": arch.num_heads,
            "vocab": arch.vocab,
            "split": split,
            "val_loss_mean": float(np.mean(losses)),
            "val_loss_std": float(np.std(losses)),
            "val_ppl_mean": float(np.mean(ppls)),
            "val_ppl_std": float(np.std(ppls)),
            "val_bpb_mean": float(np.mean(bpbs)),
            "val_bpb_std": float(np.std(bpbs)),
        }
        if on_row is not None:
            on_row(row)          # durable incremental write before continuing
        rows.append(row)
    return rows
