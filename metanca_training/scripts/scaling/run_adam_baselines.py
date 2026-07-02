# metanca_training/scripts/scaling/run_adam_baselines.py
"""Train each grid architecture with Adam TO CONVERGENCE (early stopping); save a
per-arch val-performance table.

Each network is trained until its validation loss stops improving (patience epochs),
and the metrics recorded are those at the best-val-loss epoch — so the baseline reflects
each architecture's best achievable performance, a fair reference for MetaNCA.
"""

import argparse
import json
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from metanca_training._loss import compute_loss
from metanca_training.data_utils import get_fashion_mnist_datasets, prepare_batches
from metanca_training.regular_net_functions import _unpack_model_state, evaluate_reg_net_batched
from metanca_training.scaling.arch_grid import arch_id, build_grid, build_mlp

logger = logging.getLogger(__name__)


def _val_loss(model, params, val_batches) -> float:
    params, _ = _unpack_model_state(params)  # accept raw pytree or packed state
    apply_fn = lambda p, xx: model.apply({"params": p}, xx)
    total, n = 0.0, val_batches[0].shape[0]
    for i in range(n):
        x = val_batches[0][i].astype(jnp.float32)
        total += float(compute_loss(x, val_batches[1][i], val_batches[2][i], params, apply_fn))
    return total / max(n, 1)


def _masked_ce(logits, labels, mask):
    ce = optax.softmax_cross_entropy(logits, labels)
    return jnp.sum(ce * mask.squeeze(-1)) / jnp.maximum(jnp.sum(mask), 1)


def train_to_convergence(model, train_b, val_b, key, lr=1e-3, patience=10,
                         max_epochs=1000, min_delta=1e-4):
    """Adam-train until val loss stops improving; return (best_params, best_val_loss, n_epochs)."""
    tX, tY, tM = train_b
    params = model.init(key, tX[0].astype(jnp.float32))["params"]
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def train_step(params, opt_state, x, y, m):
        def loss_fn(p):
            return _masked_ce(model.apply({"params": p}, x.astype(jnp.float32)), y, m)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    n_tb = tX.shape[0]
    best_vl, best_params, best_epoch, no_improve = float("inf"), params, 0, 0
    for epoch in range(max_epochs):
        for i in range(n_tb):
            params, opt_state, _ = train_step(params, opt_state, tX[i], tY[i], tM[i])
        vl = _val_loss(model, params, val_b)
        if vl < best_vl - min_delta:
            best_vl, best_params, best_epoch, no_improve = vl, params, epoch, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return best_params, best_vl, best_epoch + 1


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    p = argparse.ArgumentParser()
    p.add_argument("--patience", type=int, default=10, help="early-stopping patience (epochs)")
    p.add_argument("--max-epochs", type=int, default=1000, help="hard cap on epochs per arch")
    p.add_argument("--min-delta", type=float, default=1e-4, help="min val-loss improvement")
    p.add_argument("--out", type=str, default="results/adam_baselines_fashion_mnist.json")
    p.add_argument("--num-shards", type=int, default=1,
                   help="split the grid across N processes (run one per GPU)")
    p.add_argument("--shard-index", type=int, default=0, help="this process's shard [0, num_shards)")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    key = jax.random.key(0)
    X, y, tr, va = get_fashion_mnist_datasets(key)
    train_b, val_b = prepare_batches(X, y, tr, va, batch_size=512)

    # varying grid (121 archs); contains every fixed3 arch as its depth-3 slice
    grid = build_grid("varying")
    # shard across GPUs: strided slice keeps depth/size mix balanced per shard
    grid = grid[args.shard_index :: args.num_shards]
    if args.smoke:
        grid = grid[:3]
        args.patience, args.max_epochs = 1, 2

    results: dict[str, dict] = {}
    for i, widths in enumerate(grid):
        key, mk = jax.random.split(key)
        model = build_mlp(widths, 10)
        best_params, best_vl, n_epochs = train_to_convergence(
            model, train_b, val_b, mk,
            patience=args.patience, max_epochs=args.max_epochs, min_delta=args.min_delta,
        )
        val_acc = float(evaluate_reg_net_batched(model, best_params, val_b))
        train_acc = float(evaluate_reg_net_batched(model, best_params, train_b))
        results[arch_id(widths)] = {
            "val_loss": best_vl, "val_acc": val_acc,
            "train_acc": train_acc, "depth": len(widths), "epochs": n_epochs,
        }
        logger.info("[%d/%d] %s val_acc=%.4f val_loss=%.4f (converged @ %d epochs)",
                    i + 1, len(grid), arch_id(widths), val_acc, best_vl, n_epochs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    logger.info("wrote %d arch baselines to %s", len(results), out)


if __name__ == "__main__":
    main()
