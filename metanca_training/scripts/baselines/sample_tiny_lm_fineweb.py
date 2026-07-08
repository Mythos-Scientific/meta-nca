"""Sample from a TinyCausalLM fineweb checkpoint saved by train_tiny_lm_fineweb.py.

Restores params + model_config from the orbax checkpoint (best val_loss step by
default), then decodes autoregressively with temperature/top-k sampling and prints
detokenized text via the fineweb SentencePiece model.

Example:
    .venv/bin/python regular_network_baselines/scripts/sample_tiny_lm_fineweb.py \
        --checkpoint-dir regular_net_checkpoints/tiny-lm-fineweb_d512_h4_l9_mlp1024_ctx1024_bs512_<ts> \
        --prompt "The history of" --num-samples 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import sentencepiece as spm

from metanca.nn import TinyCausalLM

DEFAULT_TOKENIZER = "/home/dan/Projects/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample from a fineweb TinyCausalLM checkpoint.")
    p.add_argument(
        "--checkpoint-dir",
        required=True,
        help="run directory containing numbered orbax checkpoint steps",
    )
    p.add_argument(
        "--step", type=int, default=None, help="checkpoint step to load (default: best val_loss)"
    )
    p.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER)
    p.add_argument("--prompt", default="The")
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_checkpoint(checkpoint_dir: Path, step: int | None):
    mngr = ocp.CheckpointManager(str(checkpoint_dir.resolve()))
    if step is None:
        # pick lowest val_loss among retained steps (metrics live in each step's metadata)
        steps = mngr.all_steps()
        if not steps:
            raise FileNotFoundError(f"no checkpoint steps under {checkpoint_dir}")
        metrics = {s: mngr.metrics(s) for s in steps}
        def _loss(m):
            if "neg_val_loss" in m:
                return -m["neg_val_loss"]
            return m.get("val_loss")

        with_loss = {s: _loss(m) for s, m in metrics.items() if m and _loss(m) is not None}
        step = min(with_loss, key=with_loss.get) if with_loss else max(steps)
    # Restore via abstract shapes so arrays land on the CURRENT default device;
    # a bare StandardRestore() replays the saved sharding (cuda:0) and fails on CPU.
    state_meta = mngr.item_metadata(step)["state"]
    device = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    abstract = jax.tree.map(
        lambda m: jax.ShapeDtypeStruct(m.shape, m.dtype, sharding=device),
        state_meta,
        is_leaf=lambda x: hasattr(x, "shape") and hasattr(x, "dtype"),
    )
    restored = mngr.restore(
        step,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract),
            metadata=ocp.args.JsonRestore(),
        ),
    )
    return step, restored["state"]["params"], restored["metadata"]


def main() -> None:
    args = parse_args()
    step, params, metadata = load_checkpoint(Path(args.checkpoint_dir), args.step)
    cfg = metadata["model_config"]
    print(f"loaded step {step} (epoch {metadata.get('epoch')}) config={cfg}")

    model = TinyCausalLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        mlp_dim=cfg["mlp_dim"],
        max_seq_len=cfg["max_seq_len"],
        compute_dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
    )
    ctx = cfg["max_seq_len"]

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_model)
    prompt_ids = sp.encode(args.prompt)
    if len(prompt_ids) >= ctx:
        raise ValueError(f"prompt is {len(prompt_ids)} tokens; must be < {ctx}")

    @jax.jit
    def next_logits(tokens: jax.Array, index: jax.Array) -> jax.Array:
        # tokens: [1, ctx] fixed-size buffer; causal mask makes positions > index irrelevant
        logits = model.apply({"params": params}, tokens)
        return logits[0, index - 1, :].astype(jnp.float32)

    key = jax.random.key(args.seed)
    for sample_idx in range(args.num_samples):
        tokens = jnp.zeros((1, ctx), dtype=jnp.int32)
        tokens = tokens.at[0, : len(prompt_ids)].set(jnp.asarray(prompt_ids, dtype=jnp.int32))
        length = len(prompt_ids)
        generated = list(prompt_ids)

        for _ in range(min(args.max_new_tokens, ctx - length)):
            logits = next_logits(tokens, jnp.asarray(length)) / max(args.temperature, 1e-6)
            if args.top_k > 0:
                kth = jnp.sort(logits)[-args.top_k]
                logits = jnp.where(logits < kth, -jnp.inf, logits)
            key, sk = jax.random.split(key)
            nxt = int(jax.random.categorical(sk, logits))
            generated.append(nxt)
            tokens = tokens.at[0, length].set(nxt)
            length += 1

        text = sp.decode(generated)
        print(f"\n=== sample {sample_idx + 1} (T={args.temperature}, top_k={args.top_k}) ===")
        print(text)


if __name__ == "__main__":
    main()
