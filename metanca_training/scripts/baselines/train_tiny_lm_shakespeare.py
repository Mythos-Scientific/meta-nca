"""Adam baseline: train TinyCausalLM on Shakespeare (10k BPE) directly with backprop.

Mirrors the metaNCA setup (same model dims, same data path + tokenizer, same masked
cross-entropy loss) so the resulting train/val loss + perplexity are a fair reference
point for what the learned local-rule optimizer is being compared against.

Example:
    python train_tiny_lm_shakespeare.py \
        --text-path /home/dan/Projects/monorepo/metanca_training/data/shakespeare/input.txt \
        --tokenizer-model /home/dan/Projects/monorepo/metanca_training/data/shakespeare/shakespeare_10k_bpe.model
"""

from __future__ import annotations

import argparse
import time
from typing import Sequence

import jax
import jax.numpy as jnp
import optax
import wandb
from flax.training.train_state import TrainState

from metanca.nn import TinyCausalLM
from metanca_training._loss import masked_sparse_softmax_cross_entropy
from metanca_training.baseline_lm_data import (
    batch_source_num_batches,
    get_lm_text_datasets,
    iterate_batches,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Adam baseline for TinyCausalLM on Shakespeare 10k BPE."
    )
    p.add_argument("--text-path", required=True)
    p.add_argument("--tokenizer-model", required=True)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--mlp-dim", type=int, default=512)
    p.add_argument("--context-length", type=int, default=60)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-epochs", type=int, default=60)
    p.add_argument("--log-every-n-steps", type=int, default=50)
    p.add_argument("--wandb-project", default="metanca")
    p.add_argument("--run-name", default=None)
    return p


def mean_perplexity(loss_sum: float, tokens: int) -> tuple[float, float]:
    mean = loss_sum / tokens if tokens else 0.0
    return mean, float(jnp.exp(jnp.asarray(mean, jnp.float32)))


@jax.jit
def train_step(state, x, y, mask):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, x)
        return masked_sparse_softmax_cross_entropy(logits, y, mask)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss, jnp.sum(mask)


@jax.jit
def eval_step(state, x, y, mask):
    logits = state.apply_fn({"params": state.params}, x)
    return masked_sparse_softmax_cross_entropy(logits, y, mask), jnp.sum(mask)


def run_epoch(state, batches, *, training: bool, global_step: int):
    loss_sum = 0.0
    tokens = 0
    for x_np, y_np, m_np in iterate_batches(batches):
        x = jnp.asarray(x_np, jnp.int32)
        y = jnp.asarray(y_np, jnp.int32)
        m = jnp.asarray(m_np, bool)
        if training:
            state, loss, tc = train_step(state, x, y, m)
            global_step += 1
        else:
            loss, tc = eval_step(state, x, y, m)
        tc = int(tc)
        loss_sum += float(loss) * tc
        tokens += tc
    mean, ppl = mean_perplexity(loss_sum, tokens)
    return state, mean, ppl, global_step


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_name = args.run_name or time.strftime("adam-shakespeare-%Y%m%d-%H%M%S", time.gmtime())
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    train_batches, val_batches = get_lm_text_datasets(
        rand_key=jax.random.key(0),
        text_path=args.text_path,
        context_length=args.context_length,
        batch_size=args.batch_size,
        val_split=args.val_split,
        tokenizer_model=args.tokenizer_model,
    )

    model = TinyCausalLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_dim=args.mlp_dim,
        max_seq_len=args.context_length,
        compute_dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    params = model.init(jax.random.key(0), jnp.zeros((1, args.context_length), jnp.int32))["params"]
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(
        f"model params: {n_params:,} | train batches: {batch_source_num_batches(train_batches)} "
        f"| val batches: {batch_source_num_batches(val_batches)}",
        flush=True,
    )

    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(args.lr))

    global_step = 0
    best_val = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        state, tr_loss, tr_ppl, global_step = run_epoch(
            state, train_batches, training=True, global_step=global_step
        )
        _, va_loss, va_ppl, _ = run_epoch(
            state, val_batches, training=False, global_step=global_step
        )
        best_val = min(best_val, va_loss)
        print(
            f"epoch {epoch:3d} | train_loss={tr_loss:.4f} train_ppl={tr_ppl:.2f} "
            f"| val_loss={va_loss:.4f} val_ppl={va_ppl:.2f} | best_val={best_val:.4f}",
            flush=True,
        )
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": tr_loss,
                "train/perplexity": tr_ppl,
                "val/loss": va_loss,
                "val/perplexity": va_ppl,
                "best_val/loss": best_val,
            },
            step=global_step,
        )
    wandb.finish()


if __name__ == "__main__":
    main()
