import jax
import jax.extend.core as core
import jax.numpy as jnp
import networkx as nx
import pytest

from metanca.functional import flatten_params
from metanca.neighbors import (
    build_compute_graph,
    build_parameter_neighbor_graphs,
    get_barrier_ops,
    get_nonbarrier_ops,
)
from metanca.neighbors._convert_parameter_graph import convert_parameter_graph
from metanca.nn import MultiLayerPerceptron, forward_factory


@pytest.fixture
def barrier_ops() -> set[str]:
    return set(get_barrier_ops())


@pytest.fixture
def nonbarrier_ops() -> set[str]:
    return set(get_nonbarrier_ops())


@pytest.fixture
def mlp() -> MultiLayerPerceptron:
    return MultiLayerPerceptron(layer_specs=[4, 8, 2], use_bias=True)


@pytest.fixture
def dummy_input():
    input_dim = 4
    x = jnp.ones((1, input_dim))
    return x


@pytest.fixture
def params(mlp: MultiLayerPerceptron, dummy_input: jax.Array) -> dict:
    key = jax.random.key(0)
    return mlp.init(key, dummy_input)


@pytest.fixture
def flattened_params(params: dict) -> list[tuple[str, jax.Array]]:
    return flatten_params(params["params"])


@pytest.fixture
def jaxpr(mlp: MultiLayerPerceptron, params: dict, dummy_input: jax.Array) -> core.Jaxpr:
    forward_fn = forward_factory(mlp, params)
    jaxpr = jax.make_jaxpr(forward_fn)(dummy_input).jaxpr
    return jaxpr


@pytest.fixture
def names_vars_and_params(
    jaxpr: core.Jaxpr, flattened_params: list[tuple[str, jax.Array]]
) -> list[tuple[str, core.Var, jax.Array]]:
    return [(name, var, param) for var, (name, param) in zip(jaxpr.constvars, flattened_params)]


@pytest.fixture
def compute_graph(jaxpr: core.Jaxpr) -> nx.DiGraph:
    return build_compute_graph(jaxpr)


@pytest.fixture
def parameter_graphs(
    compute_graph: nx.DiGraph, barrier_ops: set[str], nonbarrier_ops: set[str]
) -> tuple[nx.DiGraph, nx.DiGraph]:
    return build_parameter_neighbor_graphs(
        compute_graph, barrier_ops=barrier_ops, nonbarrier_ops=nonbarrier_ops
    )


def test__convert_parameter_graph(
    parameter_graphs: tuple[nx.DiGraph, nx.DiGraph],
    names_vars_and_params: list[tuple[str, core.Var, jax.Array]],
):
    for graph in parameter_graphs:
        adj = convert_parameter_graph(graph, names_vars_and_params)
        assert adj
