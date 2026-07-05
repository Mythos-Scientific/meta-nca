"""The shared hidden-state initializer is provisioned by probing the LARGEST LLM
arch in the grid (LLMArch(128, 4, 10000, 4)); this asserts it also covers building
tasknets for the grid's extreme corners without error (may take a few CPU-minutes:
building/tracing the largest arch's TaskNet -- with its 10000-row embedding/output
projection -- is the real coverage guarantee, so its dims are kept as-is rather than
shrunk for speed).

If this test's wall-clock time grows past ~15 minutes on CPU (e.g. after a further
vocab increase), skip it via the METANCA_SKIP_SLOW_LLM_INIT_COVERAGE env var rather
than shrinking the probed arch, since the whole point is to exercise the exact
grid-max build."""

import os

import jax
import jax.numpy as jnp
import pytest
from pathlib import Path
from hydra import compose, initialize_config_dir

import metanca
from metanca_training._hydra_configs import register_configs
from metanca_training.scaling.llm_grid import LLMArch, build_tiny_lm

CONFIG_DIR = str((Path(__file__).parents[3] / "configs").resolve())
CTX = 16


@pytest.mark.skipif(
    os.environ.get("METANCA_SKIP_SLOW_LLM_INIT_COVERAGE") == "1",
    reason="grid-max (10000-row embedding) TaskNet build opted out via env var",
)
def test_grid_max_initializer_covers_extreme_archs():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config",
                      overrides=["dataset=fashion_mnist", "wandb=disabled",
                                 "checkpoint.run_name=pytest_llm_initializer_coverage",
                                 f"dataset.input_shape=[{CTX}]"])
    key = jax.random.key(0)
    probe_key, build_key_a, build_key_b = jax.random.split(key, 3)

    largest = LLMArch(128, 4, 10000, 4)
    probe_tasknet = metanca.TaskNet.build(
        model=build_tiny_lm(largest, CTX),
        input_shape=(CTX,),
        key=probe_key,
        n_spatial_dims=0,
        d_neuron=cfg.positional_encoding.d_neuron,
        d_spatial=cfg.positional_encoding.d_spatial,
        d_layer=cfg.positional_encoding.d_layer,
        dummy_input_dtype=jnp.int32,
    )
    shared = (probe_tasknet.hidden_state_initializer, probe_tasknet.hidden_dim)

    for arch, bkey in ((LLMArch(32, 4, 10000, 1), build_key_a),
                       (LLMArch(128, 16, 10000, 2), build_key_b)):
        tasknet = metanca.TaskNet.build(
            model=build_tiny_lm(arch, CTX),
            input_shape=(CTX,),
            key=bkey,
            n_spatial_dims=0,
            d_neuron=cfg.positional_encoding.d_neuron,
            d_spatial=cfg.positional_encoding.d_spatial,
            d_layer=cfg.positional_encoding.d_layer,
            shared_initializer=shared,
            dummy_input_dtype=jnp.int32,
        )
        assert tasknet.hidden_dim == probe_tasknet.hidden_dim
