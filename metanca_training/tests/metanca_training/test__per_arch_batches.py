import numpy as np
import jax, jax.numpy as jnp
import wandb
from pathlib import Path
from hydra import compose, initialize_config_dir

from metanca_training import train_metanca
from metanca_training._hydra_configs import register_configs
from metanca_training._train_metanca import _infer_dummy_input_dtype
from metanca_training.scaling.llm_grid import LLMArch, build_tiny_lm

CONFIG_DIR = str((Path(__file__).parents[2] / "configs").resolve())


def test_infer_dummy_input_dtype():
    """uint8 image batches get promoted to float32 before the model sees them
    (see `_maybe_promote_image_batch`), so the tracing dummy must be float32
    too — an int32 dummy would insert a spurious convert_element_type node
    into the traced compute graph. Non-uint8 integer dtypes (token ids) map
    to int32; floats stay float32."""
    assert _infer_dummy_input_dtype(jnp.uint8) == jnp.float32
    assert _infer_dummy_input_dtype(jnp.int32) == jnp.int32
    assert _infer_dummy_input_dtype(jnp.float32) == jnp.float32


def _fake_lm_batches(vocab, n_batches=2, bs=2, ctx=16, seed=0):
    r = np.random.default_rng(seed)
    X = r.integers(0, vocab, (n_batches, bs, ctx)).astype(np.int32)
    Y = r.integers(0, vocab, (n_batches, bs, ctx)).astype(np.int32)
    M = np.ones((n_batches, bs, ctx, 1), dtype=bool)
    return X, Y, M


def test_dict_mode_two_vocabs_smoke():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=[
            "dataset=fashion_mnist", "wandb=disabled",       # dataset fields unused in dict-mode
            "training.num_metaepochs=1", "training.update_step_scheduler_max_steps=1",
            "training.early_stopping_enabled=false", "checkpoint.run_name=pytest_lm_smoke",
            "logging.log_every_n_metaepochs=1", "dataset.input_shape=[16]",
        ])
    wandb.init(mode="disabled")
    tb = {512: _fake_lm_batches(512), 1024: _fake_lm_batches(1024, seed=1)}
    vb = {512: _fake_lm_batches(512, seed=2), 1024: _fake_lm_batches(1024, seed=3)}
    models = [build_tiny_lm(LLMArch(32, 2, 512), 16), build_tiny_lm(LLMArch(32, 2, 1024), 16)]
    test_model = build_tiny_lm(LLMArch(32, 4, 1024), 16)
    params, tvars = train_metanca(
        tb, vb, cfg=cfg, models=models, test_model=test_model,
        arch_keys=[512, 1024, 1024],
    )
    assert params is not None and hasattr(tvars, "test_tasknet")
