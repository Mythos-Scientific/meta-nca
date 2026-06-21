import jax
import jax.extend.core as core
import jax.numpy as jnp
import jax.random
import networkx as nx
import pytest

from metanca.functional import flatten_params
from metanca.neighbors._parameter_graph import build_compute_graph, build_parameter_neighbor_graphs
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


@pytest.fixture
def compute_graph(jaxpr) -> nx.DiGraph:
    return build_compute_graph(jaxpr)


@pytest.fixture
def vars_params_and_names(
    mlp: tuple[MultiLayerPerceptron, dict, jax.Array], jaxpr: core.Jaxpr
) -> list[tuple[core.Var, jax.Array, str]]:
    _, params, _ = mlp
    flattened_params = flatten_params(params["params"])
    return [(var, param, name) for var, (name, param) in zip(jaxpr.constvars, flattened_params)]


def test_build_parameter_neighbor_graphs_mlp(
    mlp: tuple[MultiLayerPerceptron, dict, jax.Array],
    compute_graph: nx.DiGraph,
    vars_params_and_names: list[tuple[core.Var, jax.Array, str]],
):

    model, _, _ = mlp
    var2name = {var: name for var, _, name in vars_params_and_names}

    forward_neighbors, backward_neighbors = build_parameter_neighbor_graphs(
        compute_graph,
    )
    # in an mlp, the forward neighbors are:
    # layer_[i]_kernel -> layer_[i-1]_bias
    # layer_[i]_bias -> layer_[i-1]_kernel
    # layer_[i]_kernel -> layer_[i]_bias
    # layer_[i]_bias -> layer_[i]_kernel

    expected_forward_neighbor_edges = []
    expected_forward_primitives: dict[tuple[str, str], str] = {}
    for layer_idx in range(len(model.layer_specs)):
        expected_forward_neighbor_edges.append(
            (f"Dense_{layer_idx}.kernel", f"Dense_{layer_idx}.bias")
        )
        expected_forward_primitives[(f"Dense_{layer_idx}.kernel", f"Dense_{layer_idx}.bias")] = (
            "dot_general"
        )

        expected_forward_neighbor_edges.append(
            (f"Dense_{layer_idx}.bias", f"Dense_{layer_idx}.kernel")
        )
        expected_forward_primitives[(f"Dense_{layer_idx}.bias", f"Dense_{layer_idx}.kernel")] = (
            "add"
        )

        if layer_idx > 0:
            expected_forward_neighbor_edges.append(
                (f"Dense_{layer_idx}.kernel", f"Dense_{layer_idx-1}.kernel")
            )
            expected_forward_primitives[
                (f"Dense_{layer_idx}.kernel", f"Dense_{layer_idx-1}.kernel")
            ] = "dot_general"

            expected_forward_neighbor_edges.append(
                (f"Dense_{layer_idx}.kernel", f"Dense_{layer_idx-1}.bias")
            )
            expected_forward_primitives[
                (f"Dense_{layer_idx}.kernel", f"Dense_{layer_idx-1}.bias")
            ] = "dot_general"

    # in an mlp, the backward neighbors are:
    # layer_[i]_kernel -> layer_[i+1]_kernel
    # layer_[i]_bias -> layer_[i+1]_kernel

    expected_backward_neighbor_edges = []
    expected_backward_primitives: dict[tuple[str, str], str] = {}
    for layer_idx in range(len(model.layer_specs)):

        if layer_idx < len(model.layer_specs) - 1:
            expected_backward_primitives[
                (f"Dense_{layer_idx}.kernel", f"Dense_{layer_idx+1}.kernel")
            ] = "dot_general"

            expected_backward_neighbor_edges.append(
                (f"Dense_{layer_idx}.kernel", f"Dense_{layer_idx+1}.kernel")
            )

            expected_backward_primitives[
                (f"Dense_{layer_idx}.bias", f"Dense_{layer_idx+1}.kernel")
            ] = "dot_general"
            expected_backward_neighbor_edges.append(
                (f"Dense_{layer_idx}.bias", f"Dense_{layer_idx+1}.kernel")
            )

    expected_backward_neighbor_edges = set(expected_backward_neighbor_edges)
    expected_forward_neighbor_edges = set(expected_forward_neighbor_edges)

    forward_neighbor_edges = set([(var2name[u], var2name[v]) for u, v in forward_neighbors.edges()])
    backward_neighbor_edges = set(
        [(var2name[u], var2name[v]) for u, v in backward_neighbors.edges()]
    )
    bwd_cases = dict(nx.get_edge_attributes(backward_neighbors, "case"))
    bwd_cases = {(var2name[u], var2name[v]): case for (u, v), case in bwd_cases.items()}

    fwd_cases = dict(nx.get_edge_attributes(forward_neighbors, "case"))
    fwd_cases = {(var2name[u], var2name[v]): case for (u, v), case in fwd_cases.items()}

    assert len((forward_neighbor_edges - expected_forward_neighbor_edges)) == 0
    assert len((expected_forward_neighbor_edges - forward_neighbor_edges)) == 0

    assert len((backward_neighbor_edges - expected_backward_neighbor_edges)) == 0
    assert len((expected_backward_neighbor_edges - backward_neighbor_edges)) == 0

    forward_primitives = nx.get_edge_attributes(forward_neighbors, "primitive")
    assert len(forward_primitives) == len(expected_forward_primitives)

    for edge, (primitive, _) in forward_primitives.items():
        u_name, v_name = map(var2name.get, edge)
        primitive_name = primitive.name

        assert expected_forward_primitives[(u_name, v_name)] == primitive_name

    backward_primitives = nx.get_edge_attributes(backward_neighbors, "primitive")
    assert len(backward_primitives) == len(expected_backward_primitives)

    for edge, (primitive, _) in backward_primitives.items():
        u_name, v_name = map(var2name.get, edge)
        primitive_name = primitive.name

        assert expected_backward_primitives[(u_name, v_name)] == primitive_name
