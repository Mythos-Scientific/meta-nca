#!/usr/bin/env python
"""Offline XLA-compilation warmer for the scaling study.

Everything a scaling run compiles is deterministic ahead of time: the T-subset
(seed = 1000*T + rep), the fixed 25-arch val set (SPLIT_SEED), the update-step schedule
(n = 1..10), and the batch shapes. This script reconstructs exactly those programs and
executes each once (one batch), so with JAX_COMPILATION_CACHE_DIR set the compiled
executables land in the persistent cache. A later real run then loads them from disk
instead of recompiling — decoupling the (CPU-bound, GPU-idle) compile phase from training.

Usage (env JAX_COMPILATION_CACHE_DIR must match the one the scheduler passes to runs):
    JAX_COMPILATION_CACHE_DIR=$PWD/.jax_cache XLA_PYTHON_CLIENT_PREALLOCATE=false \
        .venv/bin/python metanca_training/scripts/scaling/warm_cache.py \
        --ablation fixed3 --T 16 --rep 0 [--steps 1,2,3,4,5,6,7,8,9,10] [--skip-eval]

Notes:
- Run it on the SAME machine (and GPU count) as the target run: multi-GPU runs compile
  per-arch programs with archs staged to device i % ndevices; this script reproduces that
  by calling the same train-step entry points.
- Compilation is CPU-bound; avoid running it while another process on the same box is
  itself compiling (they contend for CPU). Running alongside a GPU-bound training run is
  fine (light GPU use from autotuning only).
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

import jax  # noqa: E402
import wandb  # noqa: E402
from train import load_dataset  # noqa: E402
from run_scaling import SPLIT_SEED, build_cfg  # noqa: E402

from metanca_training._calculate_metanca_gradients import stage_data_to_devices  # noqa: E402
from metanca_training._metanca_train_step import (  # noqa: E402
    metanca_train_step, metanca_train_step_multi_gpu,
)
from metanca_training._train_metanca import before_metanca_training  # noqa: E402
from metanca_training._validation_step import metanca_validation_step  # noqa: E402
from metanca_training.callbacks import CallbackRunner, create_accuracy_callback  # noqa: E402
from metanca_training.data_utils import promote_image_batch  # noqa: E402
from metanca_training.scaling.arch_grid import (  # noqa: E402
    N_VAL, arch_id, build_grid, build_mlp, sample_subset, split_grid,
)
from metanca_training.scaling.hidden_state import grid_hidden_state_initializer  # noqa: E402

logger = logging.getLogger("warm_cache")


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", choices=["fixed3", "varying"], default="fixed3")
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--steps", type=lambda s: [int(x) for x in s.split(",")],
                   default=list(range(1, 11)), help="n_update_steps values to warm (train)")
    p.add_argument("--eval-steps", type=int, default=10)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()

    if not os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        logger.warning("JAX_COMPILATION_CACHE_DIR is NOT set — compilations will not persist!")

    seed = 1000 * args.T + args.rep
    run_name = f"warm_{args.ablation}_T{args.T}_rep{args.rep}"
    cfg = build_cfg(args.ablation, run_name, 1, seed)
    wandb.init(mode="disabled")

    grid = build_grid(args.ablation)
    pool, val_archs = split_grid(grid, N_VAL[args.ablation], seed=SPLIT_SEED)
    train_archs = sample_subset(pool, args.T, seed=seed)
    n_classes = int(cfg.dataset.output_shape)

    key = jax.random.key(seed)
    train_batches, val_batches = load_dataset(cfg, key)
    batch = (
        promote_image_batch(train_batches[0][0]),
        train_batches[1][0],
        train_batches[2][0],
    )
    one_val_batch = tuple(v[:1] for v in val_batches)

    shared_init = grid_hidden_state_initializer(
        input_dim=int(cfg.dataset.input_shape[0]),
        d_neuron=cfg.positional_encoding.d_neuron,
        d_layer=cfg.positional_encoding.d_layer,
        d_spatial=cfg.positional_encoding.d_spatial,
    )
    models = [build_mlp(w, n_classes) for w in train_archs]
    tvars = before_metanca_training(
        cfg, models=models, test_model=build_mlp(val_archs[0], n_classes),
        rand_key=key, shared_initializer=shared_init,
    )
    tasknets = [tn.reset(k) for tn, k in
                zip(tvars.tasknets, jax.random.split(key, len(tvars.tasknets)))]
    adjs = tuple(tn.adj for tn in tasknets)
    pnames = tuple(tuple(tn.iter_param_names()) for tn in tasknets)
    apply_fns = tuple(tn.model.apply for tn in tasknets)
    tdl = [(tn.params, tn.hidden_states, tn.positional_encodings) for tn in tasknets]

    devices = jax.devices()
    multi = len(devices) > 1
    logger.info("warming %s T=%d rep=%d on %d device(s); train n=%s eval n=%d",
                args.ablation, args.T, args.rep, len(devices), args.steps, args.eval_steps)

    batches = tuple([batch] * len(tdl))

    if not args.skip_train:
        if multi:
            tdl_staged = stage_data_to_devices(tdl, devices)
            lr_per_dev = [jax.device_put(tvars.local_rule_params, d) for d in devices]
        for n in args.steps:
            t0 = time.time()
            if multi:
                out = metanca_train_step_multi_gpu(
                    batches=batches, local_rule_net_apply=tvars.local_rule_net_apply,
                    local_rule_params=tvars.local_rule_params,
                    lr_params_per_device=lr_per_dev, rand_key=key,
                    tasknet_data_list=tdl_staged, adjs=adjs, tasknet_param_names=pnames,
                    tasknet_apply_fns=apply_fns, n_update_steps=n,
                    hidden_dim=tvars.hidden_dim, optimizer=tvars.optimizer,
                    optimizer_state=tvars.optimizer_state, devices=devices,
                    weight_transformer_dropout=cfg.training.weight_transformer_dropout,
                    prop_cells_updated=cfg.training.prop_cells_updated,
                    grad_clip_norm=cfg.training.grad_clip_norm, n_spatial_dims=0,
                )
            else:
                out = metanca_train_step(
                    batches=batches, local_rule_net_apply=tvars.local_rule_net_apply,
                    local_rule_params=tvars.local_rule_params, rand_key=key,
                    tasknet_data_list=tdl, adjs=adjs, tasknet_param_names=pnames,
                    tasknet_apply_fns=apply_fns, n_update_steps=n,
                    hidden_dim=tvars.hidden_dim, optimizer=tvars.optimizer,
                    optimizer_state=tvars.optimizer_state,
                    weight_transformer_dropout=cfg.training.weight_transformer_dropout,
                    prop_cells_updated=cfg.training.prop_cells_updated,
                    grad_clip_norm=cfg.training.grad_clip_norm, n_spatial_dims=0,
                )
            jax.block_until_ready(jax.tree.leaves(out[1]))
            logger.info("train n=%d warmed in %.1fs", n, time.time() - t0)

    if not args.skip_eval:
        # eval programs: every train + val arch at eval-steps (matches evaluate_arch_pool)
        eval_widths = list(dict.fromkeys([tuple(w) for w in train_archs] +
                                         [tuple(w) for w in val_archs]))
        shared = (tvars.test_tasknet.hidden_state_initializer, tvars.test_tasknet.hidden_dim)
        import metanca
        for i, widths in enumerate(eval_widths, 1):
            t0 = time.time()
            tn = metanca.TaskNet.build(
                model=build_mlp(widths, n_classes), input_shape=tuple(cfg.dataset.input_shape),
                key=key, n_spatial_dims=0, d_neuron=cfg.positional_encoding.d_neuron,
                d_spatial=cfg.positional_encoding.d_spatial,
                d_layer=cfg.positional_encoding.d_layer, shared_initializer=shared,
            )
            cb = CallbackRunner.create([create_accuracy_callback()])
            metanca_validation_step(
                val_batches=one_val_batch, local_rule_net_apply=tvars.local_rule_net_apply,
                local_rule_params=tvars.local_rule_params, rand_key=key, val_tasknet=tn,
                n_update_steps=args.eval_steps, hidden_dim=tvars.hidden_dim,
                prop_cells_updated=cfg.training.prop_cells_updated,
                weight_transformer_dropout=cfg.training.weight_transformer_dropout,
                n_spatial_dims=0, callback_runner=cb,
            )
            logger.info("eval [%d/%d] %s warmed in %.1fs",
                        i, len(eval_widths), arch_id(widths), time.time() - t0)

    logger.info("cache warm complete for %s", run_name)


if __name__ == "__main__":
    main()
