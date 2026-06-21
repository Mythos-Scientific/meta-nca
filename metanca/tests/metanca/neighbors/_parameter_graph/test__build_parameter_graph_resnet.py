import jax
import jax.extend.core as core
import jax.numpy as jnp

from metanca.functional import flatten_params
from metanca.neighbors._parameter_graph import build_compute_graph, build_parameter_neighbor_graphs
from metanca.nn import forward_factory
from metanca.nn.resnet._resnet import ResidualBlock, ResNet


def test_build_parameter_neighbor_graphs_resblock_nobn():
    x = jnp.ones((1, 32, 32, 3))
    model = ResidualBlock(out_channels=8, kernel_size=3, strides=2, use_bn=False)
    params = model.init(jax.random.key(0), x, training=False)

    forward_fn = forward_factory(model, params)
    jaxpr = jax.make_jaxpr(forward_fn)(x, training=False).jaxpr
    compute_graph = build_compute_graph(jaxpr)

    flattened_params = flatten_params(params["params"])
    vars_params_and_names: list[tuple[core.Var, jax.Array, str]] = [
        (var, param, name) for var, (name, param) in zip(jaxpr.constvars, flattened_params)
    ]
    var2name = {var: name for var, _, name in vars_params_and_names}

    forward_neighbors, backward_neighbors = build_parameter_neighbor_graphs(compute_graph)

    forward_edges = {(var2name[u], var2name[v]) for u, v in forward_neighbors.edges()}
    backward_edges = {(var2name[u], var2name[v]) for u, v in backward_neighbors.edges()}

    assert len(vars_params_and_names) == len(flattened_params)
    assert len(forward_edges) > 0
    assert len(backward_edges) > 0

    # input x
    # x' = Conv1(Conv0(x))
    # x'' = Conv2(x)
    # y = x' + x''

    expected_forward_edges = [
        ("Conv_1.kernel", "Conv_0.kernel"),
        ("Conv_2.kernel", "Conv_1.kernel"),
        ("Conv_1.kernel", "Conv_2.kernel"),
    ]
    expected_backward_edges = [("Conv_0.kernel", "Conv_1.kernel")]

    assert not (set(forward_edges) - set(expected_forward_edges))  # complete coverage
    assert not (set(backward_edges) - set(expected_backward_edges))

    for expected_edge in expected_forward_edges:
        assert expected_edge in set(forward_edges)

    for expected_edge in expected_backward_edges:
        assert expected_edge in set(backward_edges)


def test_build_parameter_neighbor_graphs_resnet():
    x = jnp.ones((1, 32, 32, 3))
    model = ResNet(num_classes=10, stage_blocks=(1, 1, 1), base_channels=8, use_bn=False)
    params = model.init(jax.random.key(0), x, training=False)

    forward_fn = forward_factory(model, params)
    jaxpr = jax.make_jaxpr(forward_fn)(x).jaxpr
    compute_graph = build_compute_graph(jaxpr)

    flattened_params = flatten_params(params["params"])
    vars_params_and_names: list[tuple[core.Var, jax.Array, str]] = [
        (var, param, name) for var, (name, param) in zip(jaxpr.constvars, flattened_params)
    ]
    var2name = {var: name for var, _, name in vars_params_and_names}

    forward_neighbors, backward_neighbors = build_parameter_neighbor_graphs(compute_graph)

    forward_edges = {(var2name[u], var2name[v]) for u, v in forward_neighbors.edges()}
    backward_edges = {(var2name[u], var2name[v]) for u, v in backward_neighbors.edges()}

    assert len(vars_params_and_names) == len(flattened_params)
    assert len(forward_edges) > 0
    assert len(backward_edges) > 0

    # these test the edges we want to see between the initial conv layer and first res block
    expected_forward_edges = [
        ("Conv_0.kernel", "Conv_0.bias"),
        ("Conv_0.bias", "Conv_0.kernel"),
        ("ResidualBlock_0.Conv_0.kernel", "Conv_0.bias"),
        ("ResidualBlock_0.Conv_0.kernel", "Conv_0.kernel"),
        ("ResidualBlock_0.Conv_1.kernel", "ResidualBlock_0.Conv_0.kernel"),
        ("ResidualBlock_0.Conv_1.kernel", "Conv_0.kernel"),
        ("ResidualBlock_0.Conv_1.kernel", "Conv_0.bias"),
    ]

    expected_backward_edges = [
        ("Conv_0.kernel", "ResidualBlock_0.Conv_0.kernel"),
        ("Conv_0.bias", "ResidualBlock_0.Conv_0.kernel"),
        ("ResidualBlock_0.Conv_0.kernel", "ResidualBlock_0.Conv_1.kernel"),
    ]

    for expected_edge in expected_forward_edges:
        assert expected_edge in set(forward_edges)

    for expected_edge in expected_backward_edges:
        assert expected_edge in set(backward_edges)

    assert ("ResidualBlock_1.Conv_0.kernel", "ResidualBlock_0.Conv_1.kernel") in set(forward_edges)
    assert ("ResidualBlock_0.Conv_1.kernel", "ResidualBlock_1.Conv_0.kernel") in set(backward_edges)

    assert ("ResidualBlock_2.Conv_0.kernel", "ResidualBlock_1.Conv_2.kernel") in set(forward_edges)
    assert ("ResidualBlock_2.Conv_0.kernel", "ResidualBlock_1.Conv_2.kernel")[::-1] in set(
        backward_edges
    )

    assert ("ResidualBlock_2.Conv_0.kernel", "ResidualBlock_1.Conv_1.kernel") in set(forward_edges)
    assert ("ResidualBlock_2.Conv_0.kernel", "ResidualBlock_1.Conv_1.kernel")[::-1] in set(
        backward_edges
    )
    assert ("ResidualBlock_2.Conv_2.kernel", "ResidualBlock_1.Conv_2.kernel") in set(forward_edges)
    assert ("ResidualBlock_2.Conv_2.kernel", "ResidualBlock_1.Conv_1.kernel") in set(forward_edges)
