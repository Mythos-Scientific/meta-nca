# metanca_training/scripts/scaling/run_scaling.py
"""Run one (ablation, T, rep) meta-training + per-arch evaluation for the scaling study."""

import argparse
import json
import logging
import sys
from pathlib import Path

import jax
import wandb
from hydra import compose, initialize_config_dir

# reuse the dataset loader from the training entry point
sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
from train import load_dataset  # noqa: E402

from metanca_training import train_metanca  # noqa: E402
from metanca_training._hydra_configs import register_configs  # noqa: E402
from metanca_training.scaling.arch_grid import (  # noqa: E402
    N_VAL, WIDTHS, build_grid, build_mlp, sample_subset, split_grid,
)
from metanca_training.scaling.hidden_state import grid_hidden_state_initializer  # noqa: E402
from metanca_training.scaling.evaluate_pool import evaluate_arch_pool  # noqa: E402

CONFIG_DIR = str((Path(__file__).parents[2] / "configs").resolve())
logger = logging.getLogger(__name__)

SPLIT_SEED = 20260701  # fixed: the held-out V is identical across all T and reps


def build_cfg(ablation: str, run_name: str, metaepochs: int, seed: int):
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(
            config_name="config",
            overrides=[
                "dataset=fashion_mnist", "wandb=disabled",
                f"training.num_metaepochs={metaepochs}",
                "training.update_step_scheduler_type=increment",
                "training.update_step_scheduler_rate=100",
                "training.update_step_scheduler_max_steps=10",
                "training.early_stopping_enabled=false",
                "training.sample_pooling.enabled=false",
                "checkpoint.save_top_n=1",
                f"checkpoint.run_name={run_name}",
                f"random.seed={seed}",
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
    p.add_argument("--ablation", choices=["fixed3", "varying"], required=True)
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--metaepochs", type=int, default=1200)
    p.add_argument("--results-dir", type=str, default="results/scaling")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.metaepochs = 2

    seed = 1000 * args.T + args.rep
    run_name = f"scaling_{args.ablation}_T{args.T}_rep{args.rep}"

    run_dir = Path(args.results_dir) / args.ablation
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / f"T{args.T}_rep{args.rep}.jsonl"
    done_marker = run_dir / f"T{args.T}_rep{args.rep}.done"
    if done_marker.exists():
        logger.info("run already complete (%s); nothing to do", done_marker)
        return

    grid = build_grid(args.ablation)
    pool, val_archs = split_grid(grid, N_VAL[args.ablation], seed=SPLIT_SEED)
    train_archs = sample_subset(pool, args.T, seed=seed)
    logger.info("ablation=%s T=%d rep=%d | pool=%d val=%d train=%d",
                args.ablation, args.T, args.rep, len(pool), len(val_archs), len(train_archs))

    cfg = build_cfg(args.ablation, run_name, args.metaepochs, seed)
    # Log every scaling run to wandb (project: architecture-scaling-ablation). Use the run_name
    # as a stable id + resume="allow" so a resumed run continues the same wandb run. Smoke runs
    # stay disabled.
    wandb.init(
        project="architecture-scaling-ablation",
        name=run_name,
        id=run_name,
        resume="allow",
        mode=("disabled" if args.smoke else "online"),
        config={
            "ablation": args.ablation, "T": args.T, "rep": args.rep, "seed": seed,
            "metaepochs": args.metaepochs, "widths": list(WIDTHS),
        },
    )

    key = jax.random.key(seed)
    train_batches, val_batches = load_dataset(cfg, key)
    if args.smoke:  # keep the smoke run tiny/fast
        val_batches = tuple(v[:1] for v in val_batches)

    n_classes = int(cfg.dataset.output_shape)
    models = [build_mlp(w, n_classes) for w in train_archs]
    test_model = build_mlp(val_archs[0], n_classes)  # for in-loop monitor/logging only

    # Explicit grid-max hidden-state initializer, shared by training and eval, so every
    # train + val arch is provisioned/encoded identically regardless of the sampled T-subset.
    shared_init = grid_hidden_state_initializer(
        input_dim=int(cfg.dataset.input_shape[0]),
        d_neuron=cfg.positional_encoding.d_neuron,
        d_layer=cfg.positional_encoding.d_layer,
        d_spatial=cfg.positional_encoding.d_spatial,
    )
    # Resumes from the per-run checkpoint (local_rule_checkpoints/<run_name>) if present.
    params, tvars = train_metanca(
        train_batches, val_batches, cfg=cfg, models=models, test_model=test_model,
        shared_initializer=shared_init,
    )

    skip_ids = _existing_arch_ids(out)   # resume: don't re-evaluate finished archs
    if skip_ids:
        logger.info("resuming eval; %d archs already recorded", len(skip_ids))

    archs = [(w, "train") for w in train_archs] + [(w, "val") for w in val_archs]
    fout = out.open("a")  # append; one flushed line per arch (durable)

    def write_row(r: dict) -> None:
        fout.write(json.dumps({**r, "ablation": args.ablation, "T": args.T,
                               "rep": args.rep, "seed": seed}) + "\n")
        fout.flush()

    key, eval_key = jax.random.split(key)
    try:
        rows = evaluate_arch_pool(
            training_vars=tvars, local_rule_params=params, archs=archs,
            val_batches=val_batches, cfg=cfg, n_update_steps=10,
            n_init_samples=(1 if args.smoke else 5), rand_key=eval_key,
            skip_ids=skip_ids, on_row=write_row,
        )
    finally:
        fout.close()

    wandb.finish()
    done_marker.write_text("")   # mark run complete only after all archs are written
    logger.info("wrote %d new rows to %s (run complete)", len(rows), out)


if __name__ == "__main__":
    main()
