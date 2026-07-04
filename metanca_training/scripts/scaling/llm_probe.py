# metanca_training/scripts/scaling/llm_probe.py
"""Probe per-arch TaskNet build/train-step timing and one T=8 mixed-vocab pool steady rate
for the LLM (TinyCausalLM / Shakespeare) scaling study, then project full-run hours.

Structured like memory_probe.py but adapted to the LLM stack (dict-mode batches, TinyCausalLM
grid corners). Three cost centers are measured:
  1. Per-arch corners (LLMArch(32,2,512), LLMArch(128,4,2048), LLMArch(256,4,4096)):
     TaskNet build time, plus single-arch `metanca_train_step` compile + steady s/batch at
     n_update_steps in {1, 10}.
  2. One T=8 mixed-vocab pool: dict-mode `metanca_train_step` compile + steady s/batch at
     n_update_steps in {1, 10}, using the real Shakespeare token streams.
  3. Projections: per-metaepoch seconds at n=1/n=10, and projected full-run hours for
     T in {1,2,4,8,16,32} at --metaepochs with the increment schedule (per-arch cost scaled
     linearly in T from the T=8 measurement), plus an eval-time projection from the corner
     numbers. Prints peak device memory (the feasibility gate).
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from statistics import mean

import jax
import jax.numpy as jnp
import wandb

sys.path.insert(0, str(Path(__file__).parent))
from run_llm_scaling import LARGEST_LLM_ARCH, build_cfg  # noqa: E402 (same directory)

import metanca  # noqa: E402
from metanca_training._train_metanca import before_metanca_training  # noqa: E402
from metanca_training._metanca_train_step import metanca_train_step  # noqa: E402
from metanca_training._update_step_schedule import increment_update_step  # noqa: E402
from metanca_training.lm_data import load_multi_vocab_shakespeare  # noqa: E402
from metanca_training.scaling.llm_grid import (  # noqa: E402
    LLMArch, N_VAL_LLM, build_tiny_lm, sample_llm_subset, split_llm_grid,
)

logger = logging.getLogger(__name__)

CORNER_ARCHS = [LLMArch(32, 2, 512), LLMArch(128, 4, 2048), LLMArch(256, 4, 4096)]
T_GRID = (1, 2, 4, 8, 16, 32)
T8_SEED = 8000
BATCH_SIZE = 8


def _train_step_call(*, batches, local_rule_net_apply, local_rule_params, rand_key,
                      tasknet_data_list, adjs, tasknet_param_names, tasknet_apply_fns,
                      n_update_steps, hidden_dim, optimizer, optimizer_state, cfg):
    return metanca_train_step(
        batches=batches,
        local_rule_net_apply=local_rule_net_apply,
        local_rule_params=local_rule_params,
        rand_key=rand_key,
        tasknet_data_list=tasknet_data_list,
        adjs=adjs,
        tasknet_param_names=tasknet_param_names,
        tasknet_apply_fns=tasknet_apply_fns,
        n_update_steps=n_update_steps,
        hidden_dim=hidden_dim,
        optimizer=optimizer,
        optimizer_state=optimizer_state,
        weight_transformer_dropout=cfg.training.weight_transformer_dropout,
        prop_cells_updated=cfg.training.prop_cells_updated,
        grad_clip_norm=cfg.training.grad_clip_norm,
        n_spatial_dims=0,
    )


def time_corner(arch: LLMArch, ctx: int, cfg, key: jax.Array) -> dict:
    """TaskNet build time + single-arch train-step compile/steady at n in {1, 10}."""
    model = build_tiny_lm(arch, ctx)

    key, bk = jax.random.split(key)
    t0 = time.time()
    jax.block_until_ready(metanca.TaskNet.build(
        model=model, input_shape=(ctx,), key=bk, n_spatial_dims=0,
        d_neuron=cfg.positional_encoding.d_neuron,
        d_spatial=cfg.positional_encoding.d_spatial,
        d_layer=cfg.positional_encoding.d_layer,
        dummy_input_dtype=jnp.int32,
    ))
    build_s = time.time() - t0

    key, tk = jax.random.split(key)
    tvars = before_metanca_training(
        cfg, models=[model], test_model=model, rand_key=tk,
        shared_initializer=None, dummy_input_dtype=jnp.int32,
    )
    tasknet = tvars.tasknets[0]
    adjs = (tasknet.adj,)
    tasknet_param_names = (tuple(tasknet.iter_param_names()),)
    tasknet_apply_fns = (tasknet.model.apply,)

    key, xk, yk = jax.random.split(key, 3)
    batch = (
        jax.random.randint(xk, (BATCH_SIZE, ctx), 0, arch.vocab, dtype=jnp.int32),
        jax.random.randint(yk, (BATCH_SIZE, ctx), 0, arch.vocab, dtype=jnp.int32),
        jnp.ones((BATCH_SIZE, ctx, 1), dtype=bool),
    )

    results: dict = {"arch": arch, "build_s": build_s}
    for n in (1, 10):
        key, rk = jax.random.split(key)
        tasknet_r = tasknet.reset(rk)
        tdl = [(tasknet_r.params, tasknet_r.hidden_states, tasknet_r.positional_encodings)]
        params, opt_state, rand_key = tvars.local_rule_params, tvars.optimizer_state, key

        def _call(params, opt_state, tdl, rand_key, n=n):
            return _train_step_call(
                batches=(batch,), local_rule_net_apply=tvars.local_rule_net_apply,
                local_rule_params=params, rand_key=rand_key, tasknet_data_list=tdl,
                adjs=adjs, tasknet_param_names=tasknet_param_names,
                tasknet_apply_fns=tasknet_apply_fns, n_update_steps=n,
                hidden_dim=tvars.hidden_dim, optimizer=tvars.optimizer,
                optimizer_state=opt_state, cfg=cfg,
            )

        t0 = time.time()
        out = _call(params, opt_state, tdl, rand_key)
        jax.block_until_ready(out)
        compile_s = time.time() - t0
        tdl, _, params, opt_state, rand_key = out

        steady = []
        for _ in range(5):
            t0 = time.time()
            out = _call(params, opt_state, tdl, rand_key)
            jax.block_until_ready(out)
            steady.append(time.time() - t0)
            tdl, _, params, opt_state, rand_key = out
        results[f"n{n}_compile_s"] = compile_s
        results[f"n{n}_steady_s"] = mean(steady)
    return results


def time_pool(cfg, ctx: int, mvlm, t8_batches: int, key: jax.Array) -> dict:
    """T=8 mixed-vocab pool: dict-mode train-step compile/steady at n in {1, 10}."""
    pool, val_archs = split_llm_grid(20260701)
    train_archs = sample_llm_subset(pool, 8, seed=T8_SEED)
    models = [build_tiny_lm(a, ctx) for a in train_archs]
    test_model = build_tiny_lm(val_archs[0], ctx)

    key, pk = jax.random.split(key)
    probe_tasknet = metanca.TaskNet.build(
        model=build_tiny_lm(LARGEST_LLM_ARCH, ctx), input_shape=(ctx,), key=pk,
        n_spatial_dims=0, d_neuron=cfg.positional_encoding.d_neuron,
        d_spatial=cfg.positional_encoding.d_spatial, d_layer=cfg.positional_encoding.d_layer,
        dummy_input_dtype=jnp.int32,
    )
    shared_init = (probe_tasknet.hidden_state_initializer, probe_tasknet.hidden_dim)

    key, tk = jax.random.split(key)
    tvars = before_metanca_training(
        cfg, models=models, test_model=test_model, rand_key=tk,
        shared_initializer=shared_init, dummy_input_dtype=jnp.int32,
    )
    tasknets = tvars.tasknets
    adjs = tuple(t.adj for t in tasknets)
    tasknet_param_names = tuple(tuple(t.iter_param_names()) for t in tasknets)
    tasknet_apply_fns = tuple(t.model.apply for t in tasknets)

    key, rk = jax.random.split(key)
    reset_keys = jax.random.split(rk, num=len(tasknets))
    tasknets = [t.reset(k) for t, k in zip(tasknets, reset_keys)]
    tdl0 = [(t.params, t.hidden_states, t.positional_encodings) for t in tasknets]

    n_train_batches = int(mvlm.train[train_archs[0].vocab][0].shape[0])

    def batch_for(idx):
        i = idx % n_train_batches
        return tuple(
            (mvlm.train[a.vocab][0][i], mvlm.train[a.vocab][1][i], mvlm.train[a.vocab][2][i])
            for a in train_archs
        )

    results: dict = {"T": 8, "n_train_batches": n_train_batches}
    for n in (1, 10):
        params, opt_state, tdl, rand_key = (
            tvars.local_rule_params, tvars.optimizer_state, tdl0, key,
        )

        def _call(params, opt_state, tdl, rand_key, batches, n=n):
            return _train_step_call(
                batches=batches, local_rule_net_apply=tvars.local_rule_net_apply,
                local_rule_params=params, rand_key=rand_key, tasknet_data_list=tdl,
                adjs=adjs, tasknet_param_names=tasknet_param_names,
                tasknet_apply_fns=tasknet_apply_fns, n_update_steps=n,
                hidden_dim=tvars.hidden_dim, optimizer=tvars.optimizer,
                optimizer_state=opt_state, cfg=cfg,
            )

        t0 = time.time()
        out = _call(params, opt_state, tdl, rand_key, batch_for(0))
        jax.block_until_ready(out)
        compile_s = time.time() - t0
        tdl, _, params, opt_state, rand_key = out

        steady = []
        for i in range(t8_batches):
            t0 = time.time()
            out = _call(params, opt_state, tdl, rand_key, batch_for(i + 1))
            jax.block_until_ready(out)
            steady.append(time.time() - t0)
            tdl, _, params, opt_state, rand_key = out
        results[f"n{n}_compile_s"] = compile_s
        results[f"n{n}_steady_s"] = mean(steady)
    return results


def simulate_n_update_steps(metaepochs: int, rate: int, max_steps: int) -> list[int]:
    """Replay the increment-update-step schedule train_metanca uses, per metaepoch."""
    schedule = increment_update_step(rate, max_steps)
    n = min(1, max_steps) if max_steps is not None else 1
    ns = []
    for me in range(metaepochs):
        n = schedule(me, n)
        ns.append(n)
    return ns


def _interp_per_batch(n: int, steady1: float, steady10: float) -> float:
    """Linear interpolation in n_update_steps between the measured n=1 and n=10 rates."""
    return steady1 + (n - 1) / (10 - 1) * (steady10 - steady1)


def project(corner_results: list[dict], pool_results: dict, n_batches: int,
            metaepochs: int, rate: int, max_steps: int) -> None:
    logger.info("=== PROJECTIONS ===")
    per_arch_s1 = pool_results["n1_steady_s"] / 8
    per_arch_s10 = pool_results["n10_steady_s"] / 8
    logger.info("T=8 pool per-metaepoch: n=1 => %.2fs, n=10 => %.2fs (n_batches=%d)",
                pool_results["n1_steady_s"] * n_batches,
                pool_results["n10_steady_s"] * n_batches, n_batches)

    ns = simulate_n_update_steps(metaepochs, rate, max_steps)
    eval_per_arch_s = mean(r["build_s"] + r["n10_steady_s"] for r in corner_results)

    for T in T_GRID:
        s1_T, s10_T = per_arch_s1 * T, per_arch_s10 * T
        train_s = sum(n_batches * _interp_per_batch(n, s1_T, s10_T) for n in ns)
        n_eval_archs = T + N_VAL_LLM
        eval_s = eval_per_arch_s * n_eval_archs
        total_h = (train_s + eval_s) / 3600
        logger.info(
            "T=%2d: train %.1f h + eval %.1f h (%d archs) = %.1f h total",
            T, train_s / 3600, eval_s / 3600, n_eval_archs, total_h,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    p = argparse.ArgumentParser()
    p.add_argument("--metaepochs", type=int, default=1200)
    p.add_argument("--t8-batches", type=int, default=20,
                   help="# steady batches to average for the T=8 pool measurement")
    p.add_argument("--skip-pool", action="store_true", help="corners only, skip T=8 pool + T-grid projection")
    p.add_argument("--data-dir", type=str, default="../../data/shakespeare",
                   help="shakespeare data dir, relative to this script's directory by default")
    args = p.parse_args()

    wandb.init(mode="disabled")
    key = jax.random.key(0)

    data_dir = str((Path(__file__).parent / args.data_dir).resolve())
    mvlm = load_multi_vocab_shakespeare(data_dir, batch_size=BATCH_SIZE)
    ctx = mvlm.context_length
    n_batches = int(mvlm.train[mvlm.vocabs[0]][0].shape[0])
    logger.info("loaded shakespeare: ctx=%d vocabs=%s n_train_batches=%d", ctx, mvlm.vocabs, n_batches)

    cfg = build_cfg("llm_probe", args.metaepochs, seed=0, context_length=ctx)

    logger.info("=== CORNERS ===")
    corner_results = []
    for arch in CORNER_ARCHS:
        key, ck = jax.random.split(key)
        r = time_corner(arch, ctx, cfg, ck)
        corner_results.append(r)
        logger.info(
            "%s: build=%.3fs | n=1 compile=%.3fs steady=%.4fs/batch | "
            "n=10 compile=%.3fs steady=%.4fs/batch",
            arch, r["build_s"], r["n1_compile_s"], r["n1_steady_s"],
            r["n10_compile_s"], r["n10_steady_s"],
        )

    if not args.skip_pool:
        logger.info("=== T=8 MIXED-VOCAB POOL ===")
        key, pk = jax.random.split(key)
        pool_results = time_pool(cfg, ctx, mvlm, args.t8_batches, pk)
        logger.info(
            "T=8: n=1 compile=%.3fs steady=%.4fs/batch | n=10 compile=%.3fs steady=%.4fs/batch",
            pool_results["n1_compile_s"], pool_results["n1_steady_s"],
            pool_results["n10_compile_s"], pool_results["n10_steady_s"],
        )
        project(
            corner_results, pool_results, n_batches, args.metaepochs,
            rate=int(cfg.training.update_step_scheduler_rate),
            max_steps=int(cfg.training.update_step_scheduler_max_steps),
        )
    else:
        logger.info("--skip-pool: skipping T=8 pool measurement and T-grid projection")

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
