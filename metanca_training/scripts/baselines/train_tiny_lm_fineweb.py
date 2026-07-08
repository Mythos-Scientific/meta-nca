from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState

from metanca.nn import TinyCausalLM
from metanca_training._checkpointing import (
    build_checkpoint_metadata,
    create_checkpoint_manager,
    save_checkpoint,
)
from metanca_training._loss import masked_sparse_softmax_cross_entropy
from metanca_training.baseline_lm_data import (
    FINEWEB_VOCAB_SIZE,
    batch_source_num_batches,
    get_fineweb_datasets,
    iterate_batches,
)


@dataclass
class MetricAccumulator:
    loss_sum: float = 0.0
    token_count: int = 0
    # corpus tokens-per-byte; bpb is only correct when this is set (see
    # estimate_tokens_per_byte). None -> bpb falls back to bits/token and is
    # reported as such.
    tokens_per_byte: float | None = None

    def update(self, *, loss: float, token_count: int) -> None:
        self.loss_sum += loss * token_count
        self.token_count += token_count

    @property
    def mean_loss(self) -> float:
        if self.token_count == 0:
            return 0.0
        return self.loss_sum / self.token_count

    @property
    def perplexity(self) -> float:
        return float(jnp.exp(jnp.asarray(self.mean_loss, dtype=jnp.float32)))

    @property
    def bits_per_token(self) -> float:
        return float(jnp.asarray(self.mean_loss, dtype=jnp.float32) / jnp.log(2.0))

    @property
    def bpb(self) -> float:
        # bits/byte = (nats/token) * (tokens/byte) / ln(2); the old
        # implementation omitted tokens/byte and reported bits/token as "bpb"
        if self.tokens_per_byte is None:
            return self.bits_per_token
        return self.bits_per_token * self.tokens_per_byte


def estimate_tokens_per_byte(
    shard_path: str | Path, tokenizer_model: str, sample_tokens: int = 2_000_000
) -> float:
    """Detokenize a prefix of a shard to measure the corpus tokens/byte ratio."""
    import sentencepiece as spm

    from metanca_training.baseline_lm_data import load_fineweb_shard_tokens

    sp = spm.SentencePieceProcessor(model_file=tokenizer_model)
    tokens = np.asarray(load_fineweb_shard_tokens(shard_path)[:sample_tokens], dtype=np.int32)
    n_bytes = 0
    chunk = 100_000
    for start in range(0, len(tokens), chunk):
        n_bytes += len(sp.decode(tokens[start : start + chunk].tolist()).encode("utf-8"))
    return len(tokens) / n_bytes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TinyCausalLM on FineWeb with Adam.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--mlp-dim", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--log-every-n-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--wandb-project", default="fineweb-llm-baseline")
    parser.add_argument("--checkpoint-dir", default="regular_net_checkpoints")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-top-n", type=int, default=3)
    return parser


def create_model(
    *,
    context_length: int,
    d_model: int,
    num_heads: int,
    num_layers: int,
    mlp_dim: int,
) -> TinyCausalLM:
    return TinyCausalLM(
        vocab_size=FINEWEB_VOCAB_SIZE,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_dim=mlp_dim,
        max_seq_len=context_length,
        compute_dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
    )


def create_train_state(
    *, model: TinyCausalLM, learning_rate: float, context_length: int
) -> TrainState:
    init_tokens = jnp.zeros((1, context_length), dtype=jnp.int32)
    params = model.init(jax.random.key(0), init_tokens)["params"]
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(
    state: TrainState,
    batch_x: jax.Array,
    batch_y: jax.Array,
    batch_mask: jax.Array,
) -> tuple[TrainState, jax.Array, jax.Array]:
    token_count = jnp.sum(batch_mask)

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch_x)
        return masked_sparse_softmax_cross_entropy(logits, batch_y, batch_mask)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    next_state = state.apply_gradients(grads=grads)
    return next_state, loss, token_count


@jax.jit
def eval_step(
    state: TrainState,
    batch_x: jax.Array,
    batch_y: jax.Array,
    batch_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    logits = state.apply_fn({"params": state.params}, batch_x)
    loss = masked_sparse_softmax_cross_entropy(logits, batch_y, batch_mask)
    token_count = jnp.sum(batch_mask)
    return loss, token_count


def _format_metrics(prefix: str, metrics: MetricAccumulator) -> str:
    return (
        f"{prefix}: loss={metrics.mean_loss:.4f} "
        f"perplexity={metrics.perplexity:.4f} "
        f"bpb={metrics.bpb:.4f}"
    )


def _log_metrics(prefix: str, metrics: MetricAccumulator, *, epoch_idx: int, step: int) -> None:
    wandb.log(
        {
            f"{prefix}/loss": metrics.mean_loss,
            f"{prefix}/perplexity": metrics.perplexity,
            f"{prefix}/bpb": metrics.bpb,
            f"{prefix}/epoch": epoch_idx,
        },
        step=step,
    )


def _build_run_name(args: argparse.Namespace) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return (
        f"tiny-lm-fineweb_d{args.d_model}_h{args.num_heads}_l{args.num_layers}"
        f"_mlp{args.mlp_dim}_ctx{args.context_length}_bs{args.batch_size}_{timestamp}"
    )


def run_epoch(
    *,
    state: TrainState,
    batches,
    epoch_idx: int,
    log_every_n_steps: int,
    is_training: bool,
    global_step: int,
    tokens_per_byte: float | None = None,
) -> tuple[TrainState, MetricAccumulator, int]:
    num_batches = batch_source_num_batches(batches)
    metrics = MetricAccumulator(tokens_per_byte=tokens_per_byte)

    for step_idx, (batch_x_np, batch_y_np, batch_mask_np) in enumerate(
        iterate_batches(batches), start=1
    ):
        batch_x = jnp.asarray(batch_x_np, dtype=jnp.int32)
        batch_y = jnp.asarray(batch_y_np, dtype=jnp.int32)
        batch_mask = jnp.asarray(batch_mask_np, dtype=bool)

        if is_training:
            state, loss, token_count = train_step(state, batch_x, batch_y, batch_mask)
            global_step += 1
        else:
            loss, token_count = eval_step(state, batch_x, batch_y, batch_mask)

        loss_value = float(loss)
        token_count_value = int(token_count)
        metrics.update(loss=loss_value, token_count=token_count_value)

        if is_training:
            batch_metrics = MetricAccumulator(tokens_per_byte=tokens_per_byte)
            batch_metrics.update(loss=loss_value, token_count=token_count_value)
            _log_metrics("train", batch_metrics, epoch_idx=epoch_idx, step=global_step)

        if step_idx % log_every_n_steps == 0 or step_idx == num_batches:
            phase = "train" if is_training else "val"
            print(
                f"Epoch {epoch_idx} {phase} running avg "
                f"({step_idx}/{num_batches}): "
                f"loss={metrics.mean_loss:.4f} "
                f"perplexity={metrics.perplexity:.4f} "
                f"bpb={metrics.bpb:.4f}",
                flush=True,
            )

    phase_label = "Train" if is_training else "Val"
    print(_format_metrics(f"Epoch {epoch_idx} {phase_label}", metrics), flush=True)
    if not is_training:
        _log_metrics("val", metrics, epoch_idx=epoch_idx, step=global_step)
    return state, metrics, global_step


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    wandb.init(project=args.wandb_project, config=vars(args))
    data_dir = Path(args.data_dir)
    run_name = args.run_name or _build_run_name(args)
    checkpoint_dir = (Path(args.checkpoint_dir) / run_name).resolve()
    checkpoint_metadata = build_checkpoint_metadata(
        "adam",
        {"learning_rate": args.lr},
    )
    # meta-nca's manager keeps the N HIGHEST-metric checkpoints, so monitor -val_loss
    checkpoint_manager = create_checkpoint_manager(
        str(checkpoint_dir),
        args.save_top_n,
        "neg_val_loss",
    )
    try:
        train_batches, val_batches = get_fineweb_datasets(
            train_glob=str(data_dir / "fineweb_train_*.bin"),
            val_glob=str(data_dir / "fineweb_val_*.bin"),
            context_length=args.context_length,
            batch_size=args.batch_size,
        )
        val_shard = sorted(data_dir.glob("fineweb_val_*.bin"))[0]
        tokens_per_byte = estimate_tokens_per_byte(val_shard, args.tokenizer_model)
        print(
            f"tokens_per_byte={tokens_per_byte:.4f} (estimated from {val_shard.name})", flush=True
        )
        wandb.config.update({"tokens_per_byte": tokens_per_byte})

        model = create_model(
            context_length=args.context_length,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            mlp_dim=args.mlp_dim,
        )
        state = create_train_state(
            model=model,
            learning_rate=args.lr,
            context_length=args.context_length,
        )
        global_step = 0

        for epoch_idx in range(1, args.num_epochs + 1):
            state, _, global_step = run_epoch(
                state=state,
                batches=train_batches,
                epoch_idx=epoch_idx,
                log_every_n_steps=args.log_every_n_steps,
                is_training=True,
                global_step=global_step,
                tokens_per_byte=tokens_per_byte,
            )
            state, val_metrics, global_step = run_epoch(
                state=state,
                batches=val_batches,
                epoch_idx=epoch_idx,
                log_every_n_steps=args.log_every_n_steps,
                is_training=False,
                global_step=global_step,
                tokens_per_byte=tokens_per_byte,
            )
            save_checkpoint(
                epoch_idx,
                metrics={"neg_val_loss": -val_metrics.mean_loss},
                params=state.params,
                opt_state=state.opt_state,
                # save_checkpoint has no extra_metadata kwarg; everything rides in metadata
                metadata=checkpoint_metadata
                | {
                    "epoch": epoch_idx,
                    "global_step": global_step,
                    "model_config": {
                        "vocab_size": FINEWEB_VOCAB_SIZE,
                        "d_model": args.d_model,
                        "num_heads": args.num_heads,
                        "num_layers": args.num_layers,
                        "mlp_dim": args.mlp_dim,
                        "max_seq_len": args.context_length,
                    },
                    "run_name": run_name,
                },
                checkpoint_manager=checkpoint_manager,
            )
        checkpoint_manager.wait_until_finished()
    finally:
        checkpoint_manager.close()
        wandb.finish()


if __name__ == "__main__":
    main()
