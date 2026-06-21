import jax
import jax.extend.core as core
import jax.numpy as jnp
import pytest

from metanca.neighbors._parameter_graph import PARAMETER_OP_NAME, build_compute_graph
from metanca.nn import MultiLayerPerceptron, forward_factory

FEATURE_SPECS = [[8], [8, 2], [4, 8, 2], [4, 4, 4, 4]]


@pytest.fixture(params=FEATURE_SPECS)
def mlp(request) -> tuple[MultiLayerPerceptron, dict, jax.Array]:
    input_dim = 4
    key = jax.random.key(0)
    x = jnp.ones((1, input_dim))
    model = MultiLayerPerceptron(layer_specs=request.param, use_bias=True)

    params = model.init(key, x)
    return model, params, x


@pytest.fixture
def jaxpr(mlp: tuple[MultiLayerPerceptron, dict, jax.Array]) -> core.Jaxpr:
    model, params, x = mlp
    forward_fn = forward_factory(model, params)
    jaxpr = jax.make_jaxpr(forward_fn)(x).jaxpr
    return jaxpr


def test_build_compute_graph_two_layer(
    mlp: tuple[MultiLayerPerceptron, dict, jax.Array], jaxpr: core.Jaxpr
):

    model, _, _ = mlp
    compute_graph = build_compute_graph(jaxpr)
    parameters = [node for node in compute_graph if node.primitive.name == PARAMETER_OP_NAME]

    assert len(parameters) == len(model.layer_specs) * 2
