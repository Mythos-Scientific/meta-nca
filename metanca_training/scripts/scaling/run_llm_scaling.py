# metanca_training/scripts/scaling/run_llm_scaling.py
"""Run one (T, rep) LLM meta-training + per-arch evaluation for the Shakespeare scaling study."""

import argparse
import json
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import wandb
from hydra import compose, initialize_config_dir

import metanca
from metanca_training import train_metanca  # noqa: E402
from metanca_training._hydra_configs import register_configs  # noqa: E402
from metanca_training.lm_data import load_multi_vocab_shakespeare  # noqa: E402
from metanca_training.scaling.llm_grid import (  # noqa: E402
    D_MODELS, HEADS, LLMArch, MLP_RATIOS, VOCABS, build_tiny_lm, build_width_grid,
    nested_width_subset, sample_llm_subset, split_llm_grid, split_width_grid,
)
from metanca_training.scaling.evaluate_llm_pool import evaluate_llm_pool  # noqa: E402

CONFIG_DIR = str((Path(__file__).parents[2] / "configs").resolve())
logger = logging.getLogger(__name__)

SPLIT_SEED = 20260701  # fixed: the held-out V is identical across all T and reps
LARGEST_LLM_ARCH = LLMArch(128, 4, 10000, 4)  # spans the grid; used to provision the shared initializer
WIDTH_SUBSET_SEED = 20260706  # width study: per-rep shuffle seed base (rep-only, so T-subsets nest)


def build_cfg(run_name: str, metaepochs: int, seed: int, context_length: int,
              scheduler_rate: int = 30):
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(
            config_name="config",
            overrides=[
                "dataset=fashion_mnist", "wandb=disabled",
                f"training.num_metaepochs={metaepochs}",
                "training.update_step_scheduler_type=increment",
                f"training.update_step_scheduler_rate={scheduler_rate}",
                "training.update_step_scheduler_max_steps=10",
                "training.early_stopping_enabled=false",
                "training.sample_pooling.enabled=false",
                "checkpoint.save_top_n=1",
                f"checkpoint.run_name={run_name}",
                f"random.seed={seed}",
                f"dataset.input_shape=[{context_length}]",
            ],
        )


def _existing_arch_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                ids.add(json.loads(line)["arch_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a truncated/partial last line from a mid-write crash
    return ids


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--grid", choices=("mixed", "width"), default="mixed",
                   help="'mixed' = original d/h/r grid; 'width' = width-only study "
                        "(h=4, r=4 fixed; 24 widths; interleaved val; nested T-subsets)")
    p.add_argument("--metaepochs", type=int, default=330)
    p.add_argument("--increment-rate", type=int, default=30)
    p.add_argument("--wandb-suffix", type=str, default="v10k-m330",
                   help="appended to wandb run name/id so each study config gets fresh runs")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-eval-dmodel", type=int, default=None,
                   help="skip evaluating archs with d_model above this (e.g. 192 on 32GB "
                        "GPUs); .done is withheld so a larger-memory worker can finish them")
    p.add_argument("--results-dir", type=str, default="results/scaling")
    p.add_argument("--data-dir", type=str, default="metanca_training/data/shakespeare")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.metaepochs = 2

    seed = 1000 * args.T + args.rep
    width_study = args.grid == "width"
    run_name = f"scaling_{'llmw' if width_study else 'llm'}_T{args.T}_rep{args.rep}"
    wandb_run_name = f"{run_name}-{args.wandb_suffix}" if args.wandb_suffix else run_name

    run_dir = Path(args.results_dir) / ("llm_width" if width_study else "llm")
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / f"T{args.T}_rep{args.rep}.jsonl"
    done_marker = run_dir / f"T{args.T}_rep{args.rep}.done"
    if done_marker.exists():
        logger.info("run already complete (%s); nothing to do", done_marker)
        return

    if width_study:
        pool, val_archs = split_width_grid()
        # subset seed depends on the rep ONLY: within a rep, T-subsets are nested prefixes
        train_archs = nested_width_subset(pool, args.T, rep_seed=WIDTH_SUBSET_SEED + args.rep)
        largest_arch = build_width_grid()[-1]  # d_model=256 provisions the shared initializer
    else:
        pool, val_archs = split_llm_grid(SPLIT_SEED)
        train_archs = sample_llm_subset(pool, args.T, seed=seed)
        largest_arch = LARGEST_LLM_ARCH
    logger.info("llm grid=%s T=%d rep=%d | pool=%d val=%d train=%s",
                args.grid, args.T, args.rep, len(pool), len(val_archs),
                [a.d_model for a in train_archs] if width_study else len(train_archs))

    mvlm = load_multi_vocab_shakespeare(args.data_dir, batch_size=args.batch_size)
    ctx = mvlm.context_length

    cfg = build_cfg(run_name, args.metaepochs, seed, ctx, scheduler_rate=args.increment_rate)
    # Log every scaling run to wandb (project: architecture-scaling-ablation). Use the run_name
    # as a stable id + resume="allow" so a resumed run continues the same wandb run. Smoke runs
    # stay disabled.
    wandb.init(
        project="architecture-scaling-ablation",
        name=wandb_run_name,
        id=wandb_run_name,
        resume="allow",
        mode=("disabled" if args.smoke else "online"),
        config={
            "study": "llm_shakespeare_width" if width_study else "llm_shakespeare",
            "T": args.T, "rep": args.rep, "seed": seed, "metaepochs": args.metaepochs,
            **({"widths": [a.d_model for a in build_width_grid()],
                "train_widths": [a.d_model for a in train_archs]} if width_study else
               {"d_models": list(D_MODELS), "heads": list(HEADS), "vocabs": list(VOCABS),
                "mlp_ratios": list(MLP_RATIOS)}),
        },
    )

    train_batches, val_batches = mvlm.train, mvlm.val
    if args.smoke:  # keep the smoke run tiny/fast
        # Truncate BOTH streams: Shakespeare yields ~1300 train batches per metaepoch, which
        # makes even a 2-metaepoch smoke take hours on CPU. The smoke validates plumbing
        # (dict-mode training + 16 eval rows + durability), not training quality.
        train_batches = {v: tuple(arr[:2] for arr in b) for v, b in train_batches.items()}
        val_batches = {v: tuple(arr[:1] for arr in b) for v, b in val_batches.items()}

    models = [build_tiny_lm(a, ctx) for a in train_archs]
    test_model = build_tiny_lm(val_archs[0], ctx)  # for in-loop monitor/logging only
    arch_keys = [a.vocab for a in train_archs] + [val_archs[0].vocab]

    # Explicit grid-max hidden-state initializer, shared by training and eval, so every
    # train + val arch is provisioned/encoded identically regardless of the sampled T-subset.
    # Provisioned by probing the LARGEST arch in the grid (spans d_model/vocab extremes).
    probe_key = jax.random.key(seed)
    probe_tasknet = metanca.TaskNet.build(
        model=build_tiny_lm(largest_arch, ctx),
        input_shape=(ctx,),
        key=probe_key,
        n_spatial_dims=0,
        d_neuron=cfg.positional_encoding.d_neuron,
        d_spatial=cfg.positional_encoding.d_spatial,
        d_layer=cfg.positional_encoding.d_layer,
        dummy_input_dtype=jnp.int32,
    )
    shared_init = (probe_tasknet.hidden_state_initializer, probe_tasknet.hidden_dim)

    # In-loop metrics/checkpointing switch to perplexity/bpb (derived from loss) instead of
    # token accuracy for this LM study; the factor is the (fixed) tokenizer's val tokens/byte
    # ratio, keyed dynamically off the loader's single vocab (never hardcode 10000 here).
    (lm_vocab,) = mvlm.vocabs
    lm_val_factor = mvlm.factors[lm_vocab]["val"]

    # Resumes from the per-run checkpoint (local_rule_checkpoints/<run_name>) if present.
    params, tvars = train_metanca(
        train_batches, val_batches, cfg=cfg, models=models, test_model=test_model,
        shared_initializer=shared_init, arch_keys=arch_keys, lm_val_factor=lm_val_factor,
    )

    skip_ids = _existing_arch_ids(out)   # resume: don't re-evaluate finished archs
    if skip_ids:
        logger.info("resuming eval; %d archs already recorded", len(skip_ids))

    archs = [(a, "train") for a in train_archs] + [(a, "val") for a in val_archs]
    # biggest archs last: on memory-capped workers everything else lands durably first
    archs.sort(key=lambda t: t[0].d_model)
    n_expected = len(archs)
    if args.max_eval_dmodel is not None:
        deferred = [t for t in archs if t[0].d_model > args.max_eval_dmodel]
        archs = [t for t in archs if t[0].d_model <= args.max_eval_dmodel]
        if deferred:
            logger.info("deferring %d archs with d_model > %d to a larger-memory worker",
                        len(deferred), args.max_eval_dmodel)
    fout = out.open("a")  # append; one flushed line per arch (durable)

    def write_row(r: dict) -> None:
        fout.write(json.dumps({**r, "ablation": "llm", "T": args.T,
                               "rep": args.rep, "seed": seed}) + "\n")
        fout.flush()

    key, eval_key = jax.random.split(jax.random.key(seed))
    try:
        rows = evaluate_llm_pool(
            training_vars=tvars, local_rule_params=params, archs=archs,
            val_batches=val_batches, factors=mvlm.factors, cfg=cfg, n_update_steps=10,
            n_init_samples=(1 if args.smoke else 5), rand_key=eval_key,
            skip_ids=skip_ids, on_row=write_row,
        )
    finally:
        fout.close()

    wandb.finish()
    n_written = len(_existing_arch_ids(out))
    if n_written >= n_expected:
        done_marker.write_text("")   # complete only when EVERY arch (incl. deferred) has a row
    else:
        logger.info("run not marked done: %d/%d arch rows written (deferred archs pending)",
                    n_written, n_expected)
    logger.info("wrote %d new rows to %s (run complete)", len(rows), out)


if __name__ == "__main__":
    main()
