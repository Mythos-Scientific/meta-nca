from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from hydra.utils import instantiate
from omegaconf import ListConfig, OmegaConf

from metanca.nn import ConvMLP, MultiLayerPerceptron, ResNet

MODEL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "model"
MODEL_CONFIG_FILES = sorted(MODEL_CONFIG_DIR.glob("*.yaml"))

INPUT_SHAPES = {ConvMLP: (1, 32, 32, 3), ResNet: (1, 32, 32, 3), MultiLayerPerceptron: (1, 784)}

RAND_KEY = jax.random.key(0)


@pytest.mark.parametrize("model_cfg_path", MODEL_CONFIG_FILES, ids=lambda p: p.stem)
def test_all_model_specs_are_instantiable(model_cfg_path: Path) -> None:
    cfg = OmegaConf.create({"model": OmegaConf.load(model_cfg_path)})

    arch_cfgs = cfg.model.archs + (
        list(cfg.model.test_arch)
        if isinstance(cfg.model.test_arch, ListConfig)
        else [cfg.model.test_arch]
    )
    rks = jax.random.split(RAND_KEY, len(arch_cfgs))
    for arch_cfg, rk in zip(arch_cfgs, rks):
        model: nn.Module = instantiate(arch_cfg)
        dummy_input = jnp.ones(INPUT_SHAPES[model.__class__])
        _ = model.init(rk, dummy_input)
