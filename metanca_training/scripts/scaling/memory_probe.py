# metanca_training/scripts/scaling/memory_probe.py
"""Probe peak memory, per-metaepoch TRAINING time, and per-arch EVAL time.

Two cost centers must be projected for a real run:
  1. Training: per-metaepoch time x 12000. Measured at the SATURATED update-step count
     (constant schedule at --steps, default 10) so it reflects the dominant regime, not the
     1-step warmup the increment schedule starts in.
  2. Eval: the local rule JIT-compiles metanca_validation_step once per DISTINCT arch shape,
     so a real run pays ~(T + V) per-arch compiles. Measured per distinct arch here.
Prints peak device memory (the feasibility gate on 120GB unified memory).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import jax
import wandb

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
from train import load_dataset  # noqa: E402
from run_scaling import build_cfg  # noqa: E402 (same directory)

from metanca_training import train_metanca  # noqa: E402
from metanca_training.scaling.arch_grid import (  # noqa: E402
    N_VAL, build_grid, build_mlp, sample_subset, split_grid,
)
from metanca_training.scaling.hidden_state import grid_hidden_state_initializer  # noqa: E402
from metanca_training.scaling.evaluate_pool import evaluate_arch_pool  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--metaepochs", type=int, default=3, help="short run to time training")
    p.add_argument("--steps", type=int, default=10, help="saturated update-step count to time at")
    p.add_argument("--eval-archs", type=int, default=5, help="# distinct archs to time eval on")
    args = p.parse_args()

    grid = build_grid("fixed5")
    pool, val_archs = split_grid(grid, N_VAL["fixed5"], seed=1)
    train_archs = sample_subset(pool, args.T - 1, seed=1)
    largest = (512, 512, 512, 512, 512)  # grid max depth x width -> worst-case memory
    train_archs = [largest, *train_archs]  # force the largest arch into the pool

    cfg = build_cfg("fixed5", "mem_probe", args.metaepochs, seed=0)
    # Time at the saturated step count via a constant schedule (constant_update_step reads
    # training.num_epochs as the fixed step count).
    cfg.training.update_step_scheduler_type = "constant"
    cfg.training.num_epochs = args.steps
    wandb.init(mode="disabled")
    key = jax.random.key(0)
    train_b, val_b = load_dataset(cfg, key)
    models = [build_mlp(w, 10) for w in train_archs]

    shared_init = grid_hidden_state_initializer(
        input_dim=int(cfg.dataset.input_shape[0]),
        d_neuron=cfg.positional_encoding.d_neuron,
        d_layer=cfg.positional_encoding.d_layer,
        d_spatial=cfg.positional_encoding.d_spatial,
    )

    # --- TRAINING timing (at saturated --steps) ---
    t0 = time.time()
    params, tvars = train_metanca(
        train_b, val_b, cfg=cfg, models=models,
        test_model=build_mlp((64, 32), 10), shared_initializer=shared_init,
    )
    dt = time.time() - t0
    per_epoch = dt / args.metaepochs  # upper bound: metaepoch 0 includes ~T JIT compiles
    train_proj_h = per_epoch * 12000 / 3600

    # --- EVAL timing (per distinct arch shape; each compiles once) ---
    eval_widths, seen = [], set()
    for w in [largest, (32, 32, 32, 32, 32), *train_archs[1:], val_archs[0], val_archs[1]]:
        if w not in seen:
            seen.add(w); eval_widths.append(w)
        if len(eval_widths) >= args.eval_archs:
            break
    eval_archs = [(w, "eval") for w in eval_widths]
    key, ek = jax.random.split(key)
    te0 = time.time()
    evaluate_arch_pool(
        training_vars=tvars, local_rule_params=params, archs=eval_archs,
        val_batches=val_b, cfg=cfg, n_update_steps=args.steps,
        n_init_samples=1, rand_key=ek,
    )
    te = time.time() - te0
    per_arch = te / len(eval_archs)
    n_eval_real = args.T + N_VAL["fixed5"]
    eval_proj_h = per_arch * n_eval_real / 3600

    logger.info("=== TRAINING (T=%d, steps=%d) ===", args.T, args.steps)
    logger.info("%.1fs for %d metaepochs => %.2fs/metaepoch (upper bound; metaepoch 0 has ~%d compiles)",
                dt, args.metaepochs, per_epoch, args.T)
    logger.info("projected 12k metaepochs: %.1f h", train_proj_h)
    logger.info("=== EVAL ===")
    logger.info("%d distinct archs in %.1fs => %.1fs/arch (incl per-shape JIT compile)",
                len(eval_archs), te, per_arch)
    logger.info("projected eval for one real run (T+V=%d archs): %.1f h", n_eval_real, eval_proj_h)
    logger.info("=== TOTAL projected per T=%d run: %.1f h (train %.1f + eval %.1f) ===",
                args.T, train_proj_h + eval_proj_h, train_proj_h, eval_proj_h)
    logger.info("=== MEMORY ===")
    for d in jax.devices():
        try:
            stats = d.memory_stats()
            logger.info("device %s peak_bytes_in_use=%.1f GB",
                        d, stats.get("peak_bytes_in_use", 0) / 1e9)
        except Exception:
            pass


if __name__ == "__main__":
    main()
