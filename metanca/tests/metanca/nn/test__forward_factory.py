import jax
import jax.numpy as jnp
import pytest

from metanca.nn import ConvBlock, MultiLayerPerceptron, forward_factory

FEATURE_SPECS = [[8], [8, 2], [4, 8, 2], [4, 4, 4, 4]]

CONV_LAYER_SPEC = [
    (6, (5, 5), (4, 4), (4, 4)),
    (12, (5, 5), (4, 4), (4, 4)),
    (25, (5, 5), (4, 4), (4, 4)),
]


@pytest.fixture(params=FEATURE_SPECS)
def mlp(request) -> tuple[MultiLayerPerceptron, dict, jax.Array]:
    input_dim = 4
    key = jax.random.key(0)
    x = jnp.ones((1, input_dim))
    model = MultiLayerPerceptron(layer_specs=request.param, use_bias=True)

    params = model.init(key, x)
    return model, params, x


@pytest.fixture(params=[CONV_LAYER_SPEC])
def conv_block(request) -> tuple[ConvBlock, dict, jax.Array]:
    key = jax.random.key(0)
    data_key, param_key = jax.random.split(key, num=2)
    # imagenet dimensions
    x = jax.random.normal(data_key, shape=(4, 224, 224, 3))

    block = ConvBlock(layer_specs=CONV_LAYER_SPEC, padding="VALID", strides=1)
    params = block.init(param_key, x)
    return block, params, x


def test_foward_factory_conv_block(conv_block: tuple[ConvBlock, dict, jax.Array]):
    model, params, x = conv_block
    forward_fn = forward_factory(model, params)
    jaxpr = jax.make_jaxpr(forward_fn)(x).jaxpr

    assert len(jaxpr.invars) == 1
    assert len(jaxpr.outvars) == 1
    assert len(jaxpr.constvars) == (2 * len(model.layer_specs))


def test_forward_factory_mlp(mlp: tuple[MultiLayerPerceptron, dict, jax.Array]):
    model, params, x = mlp
    forward_fn = forward_factory(model, params)
    jaxpr = jax.make_jaxpr(forward_fn)(x).jaxpr

    assert len(jaxpr.invars) == 1
    assert len(jaxpr.outvars) == 1
    assert len(jaxpr.constvars) == (2 * len(model.layer_specs))
