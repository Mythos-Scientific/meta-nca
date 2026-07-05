import numpy as np
import jax
from pathlib import Path
from hydra import compose, initialize_config_dir

from metanca_training._hydra_configs import register_configs
from metanca_training._train_metanca import before_metanca_training
from metanca_training.scaling.llm_grid import LLMArch, build_tiny_lm
from metanca_training.scaling.evaluate_llm_pool import evaluate_llm_pool

CONFIG_DIR = str((Path(__file__).parents[3] / "configs").resolve())  # metanca_training/configs

CTX = 16


def _fake_lm_batches(vocab, n_batches=2, bs=2, ctx=CTX, seed=0):
    r = np.random.default_rng(seed)
    X = r.integers(0, vocab, (n_batches, bs, ctx)).astype(np.int32)
    Y = r.integers(0, vocab, (n_batches, bs, ctx)).astype(np.int32)
    M = np.ones((n_batches, bs, ctx, 1), dtype=bool)
    return X, Y, M


def test_evaluate_llm_pool_rows():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config",
                      overrides=["dataset=fashion_mnist", "wandb=disabled",
                                 "checkpoint.run_name=pytest_llm_eval",
                                 f"dataset.input_shape=[{CTX}]"])
    key = jax.random.key(0)

    val_batches = {512: _fake_lm_batches(512, seed=2), 1024: _fake_lm_batches(1024, seed=3)}
    factors = {512: {"val": 4.0}, 1024: {"val": 4.2}}

    train_arch = LLMArch(32, 2, 512, 1)
    val_arch = LLMArch(48, 2, 1024, 1)

    # Too slow to probe the real grid-max LLMArch(128, 4, 10000, 4) in a test; probing this
    # test's own (larger) val arch is a sufficient shared initializer for these two archs.
    probe_key, key = jax.random.split(key)
    import metanca
    import jax.numpy as jnp
    probe_tasknet = metanca.TaskNet.build(
        model=build_tiny_lm(val_arch, CTX),
        input_shape=(CTX,),
        key=probe_key,
        n_spatial_dims=0,
        d_neuron=cfg.positional_encoding.d_neuron,
        d_spatial=cfg.positional_encoding.d_spatial,
        d_layer=cfg.positional_encoding.d_layer,
        dummy_input_dtype=jnp.int32,
    )
    shared = (probe_tasknet.hidden_state_initializer, probe_tasknet.hidden_dim)

    tvars = before_metanca_training(
        cfg, models=[build_tiny_lm(train_arch, CTX)], test_model=build_tiny_lm(val_arch, CTX),
        rand_key=key, shared_initializer=shared, dummy_input_dtype=jnp.int32,
    )

    archs = [(train_arch, "train"), (val_arch, "val")]
    rows = evaluate_llm_pool(
        training_vars=tvars, local_rule_params=tvars.local_rule_params,
        archs=archs, val_batches=val_batches, factors=factors, cfg=cfg,
        n_update_steps=1, n_init_samples=1, rand_key=key,
    )
    assert len(rows) == 2
    keys = {"arch_id", "d_model", "num_heads", "vocab", "split",
            "val_loss_mean", "val_loss_std", "val_ppl_mean", "val_ppl_std",
            "val_bpb_mean", "val_bpb_std"}
    for r in rows:
        assert set(r) == keys
        for k in keys - {"arch_id", "split"}:
            assert r[k] == r[k]  # not NaN

    # resume + incremental-write behavior
    seen: list[dict] = []
    rows2 = evaluate_llm_pool(
        training_vars=tvars, local_rule_params=tvars.local_rule_params,
        archs=archs, val_batches=val_batches, factors=factors, cfg=cfg,
        n_update_steps=1, n_init_samples=1, rand_key=key,
        skip_ids={"llm_d32_h2_r1_v512"}, on_row=seen.append,
    )
    assert [r["arch_id"] for r in rows2] == ["llm_d48_h2_r1_v1024"]  # skipped d32/h2/r1/v512
    assert seen == rows2  # on_row fired per surviving arch
