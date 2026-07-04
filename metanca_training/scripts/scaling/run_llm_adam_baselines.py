# metanca_training/scripts/scaling/run_llm_adam_baselines.py
"""Train each LLM grid architecture with Adam TO CONVERGENCE (early stopping); save a
per-arch val-performance table.

Analog of run_adam_baselines.py (the MLP baseline script) for the TinyCausalLM /
Shakespeare scaling study: each (d_model, num_heads, vocab) arch is trained on its own
vocab's train batches until validation loss stops improving (patience epochs), and the
metrics recorded are those at the best-val-loss epoch -- a fair per-arch reference point
for MetaNCA.
"""

import argparse
import json
import logging
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from metanca_training._loss import masked_sparse_softmax_cross_entropy
from metanca_training.lm_data import load_multi_vocab_shakespeare
from metanca_training.scaling.llm_grid import build_llm_grid, build_tiny_lm, llm_arch_id

logger = logging.getLogger(__name__)


def _val_loss(model, params, val_b) -> float:
    X, Y, M = val_b
    total, n = 0.0, X.shape[0]
    for i in range(n):
        logits = model.apply({"params": params}, X[i])
        total += float(masked_sparse_softmax_cross_entropy(logits, Y[i], M[i]))
    return total / max(n, 1)


def train_to_convergence(model, train_b, val_b, key, lr=1e-3, patience=10,
                         max_epochs=500, min_delta=1e-4):
    """Adam-train until val loss stops improving; return (best_params, best_val_loss, n_epochs)."""
    tX, tY, tM = train_b
    params = model.init(key, tX[0])["params"]
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def train_step(params, opt_state, x, y, m):
        def loss_fn(p):
            logits = model.apply({"params": p}, x)
            return masked_sparse_softmax_cross_entropy(logits, y, m)

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
    p.add_argument("--max-epochs", type=int, default=500, help="hard cap on epochs per arch")
    p.add_argument("--min-delta", type=float, default=1e-4, help="min val-loss improvement")
    p.add_argument("--out", type=str, default="results/adam_baselines_llm_shakespeare.json")
    p.add_argument("--num-shards", type=int, default=1,
                   help="split the grid across N processes (run one per GPU)")
    p.add_argument("--shard-index", type=int, default=0, help="this process's shard [0, num_shards)")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    mvlm = load_multi_vocab_shakespeare("metanca_training/data/shakespeare", batch_size=8)
    ctx = mvlm.context_length

    grid = build_llm_grid()
    # shard across GPUs: strided slice keeps the arch mix balanced per shard
    grid = grid[args.shard_index :: args.num_shards]
    if args.smoke:
        grid = grid[:2]
        args.max_epochs = 2

    key = jax.random.key(0)
    results: dict[str, dict] = {}
    for i, arch in enumerate(grid):
        key, mk = jax.random.split(key)
        model = build_tiny_lm(arch, ctx)
        train_b = mvlm.train[arch.vocab]
        val_b = mvlm.val[arch.vocab]
        best_params, best_vl, n_epochs = train_to_convergence(
            model, train_b, val_b, mk,
            patience=args.patience, max_epochs=args.max_epochs, min_delta=args.min_delta,
        )
        val_ppl = float(jnp.exp(best_vl))
        val_bpb = best_vl * mvlm.factors[arch.vocab]["val"] / math.log(2)
        results[llm_arch_id(arch)] = {
            "val_loss": best_vl, "val_ppl": val_ppl, "val_bpb": val_bpb,
            "epochs": n_epochs, "d_model": arch.d_model, "num_heads": arch.num_heads,
            "vocab": arch.vocab,
        }
        logger.info("[%d/%d] %s val_loss=%.4f val_ppl=%.2f val_bpb=%.4f (converged @ %d epochs)",
                    i + 1, len(grid), llm_arch_id(arch), best_vl, val_ppl, val_bpb, n_epochs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    logger.info("wrote %d arch baselines to %s", len(results), out)


if __name__ == "__main__":
    main()
