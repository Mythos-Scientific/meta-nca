# metanca_training/scripts/scaling/run_adam_baselines.py
"""Train each grid architecture with Adam; save a val-performance table."""

import argparse
import json
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from metanca_training.data_utils import get_fashion_mnist_datasets, prepare_batches
from metanca_training.regular_net_functions import (
    _unpack_model_state, evaluate_reg_net_batched, init_and_train_regular_network_batched,
)
from metanca_training._loss import compute_loss
from metanca_training.scaling.arch_grid import arch_id, build_grid, build_mlp

logger = logging.getLogger(__name__)


def _val_loss(model, model_state, val_batches) -> float:
    params, _ = _unpack_model_state(model_state)   # no-bn MLP -> raw params pytree
    apply_fn = lambda p, xx: model.apply({"params": p}, xx)
    total, n = 0.0, val_batches[0].shape[0]
    for i in range(n):
        x = val_batches[0][i].astype(jnp.float32)
        total += float(compute_loss(x, val_batches[1][i], val_batches[2][i], params, apply_fn))
    return total / max(n, 1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--out", type=str, default="results/adam_baselines_fashion_mnist.json")
    p.add_argument("--num-shards", type=int, default=1,
                   help="split the grid across N processes (run one per GPU)")
    p.add_argument("--shard-index", type=int, default=0, help="this process's shard [0, num_shards)")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    key = jax.random.key(0)
    X, y, tr, va = get_fashion_mnist_datasets(key)
    train_b, val_b = prepare_batches(X, y, tr, va, batch_size=512)

    # varying grid contains all 246 archs (fixed5 subset included)
    grid = build_grid("varying")
    # shard across GPUs: strided slice keeps depth/size mix balanced per shard
    grid = grid[args.shard_index :: args.num_shards]
    if args.smoke:
        grid = grid[:3]
        args.epochs = 1

    results: dict[str, dict] = {}
    for i, widths in enumerate(grid):
        key, mk = jax.random.split(key)
        model = build_mlp(widths, 10)
        _, model_state, _, model, _, _ = init_and_train_regular_network_batched(
            mk, None, train_b, val_b, epochs=args.epochs, model=model,
            optimizer=optax.adam(1e-3),
        )
        val_acc = float(evaluate_reg_net_batched(model, model_state, val_b))
        train_acc = float(evaluate_reg_net_batched(model, model_state, train_b))
        val_loss = _val_loss(model, model_state, val_b)
        results[arch_id(widths)] = {
            "val_loss": val_loss, "val_acc": val_acc,
            "train_acc": train_acc, "depth": len(widths),
        }
        logger.info("[%d/%d] %s val_acc=%.4f val_loss=%.4f",
                    i + 1, len(grid), arch_id(widths), val_acc, val_loss)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    logger.info("wrote %d arch baselines to %s", len(results), out)


if __name__ == "__main__":
    main()
