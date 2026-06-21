import os
import time

import jax
import jax.numpy as jnp
import pytest

from metanca import update_tasknet
from metanca.functional import nested_get
from metanca.nn import LocalRuleNet, MultiLayerPerceptron
from metanca.nn._tasknet import TaskNet

CONV_LAYER_SPEC = [
    (3, (5, 5), (4, 4), (4, 4)),
    (4, (5, 5), (4, 4), (4, 4)),
    (5, (5, 5), (4, 4), (4, 4)),
]

MLP_LAYER_SPEC = [10, 20, 100]
INPUT_SHAPE = (224, 224, 3)
INPUT_SHAPE = (15,)


@pytest.fixture
def rand_key() -> jax.random.PRNGKey:
    return jax.random.key(0)


@pytest.fixture
def d_encoding() -> int:
    return 10


@pytest.fixture
def tasknet(rand_key: jax.random.PRNGKey, d_encoding: int) -> TaskNet:
    # model = ConvMLP(
    #     CONV_LAYER_SPEC,
    #     MLP_LAYER_SPEC,
    #     conv_use_bias=True,
    #     mlp_use_bias=True,
    # )
    model = MultiLayerPerceptron(layer_specs=MLP_LAYER_SPEC, use_bias=True)
    return TaskNet.build(
        model,
        INPUT_SHAPE,
        rand_key,
        n_spatial_dims=0,
        d_neuron=d_encoding,
        d_layer=d_encoding,
        d_spatial=d_encoding,
    )


@pytest.fixture
def hidden_dim(tasknet: TaskNet) -> int:
    return tasknet.hidden_dim


@pytest.fixture
def layer_widths(hidden_dim: int) -> list[int]:
    return [3 * hidden_dim + 3, 100, 200, 100, hidden_dim + 1]


@pytest.fixture
def local_rule_net_bundle(
    layer_widths: list[int], hidden_dim: int, rand_key: jax.random.PRNGKey
) -> tuple[LocalRuleNet, dict]:
    key = rand_key
    params = []
    state_dim = hidden_dim + 1  # plus weight
    local_rule_net = LocalRuleNet(
        hidden_dim=hidden_dim,
        local_rule_layer_widths=tuple(layer_widths),
        weight_transformer_dropout=0.0,
        n_spatial_dims=2,
        bias_linear_attn=False,
        bias_local_rule=True,
    )
    dummy_focus = jnp.zeros((1, 1, state_dim))
    dummy_neighbors = jnp.zeros((1, 1, state_dim))
    dummy_pos_enc = jnp.zeros((1, 1, hidden_dim))
    dummy_mask = jnp.ones((1, 1), dtype=bool)
    params = local_rule_net.init(
        key,
        dummy_focus,
        dummy_focus,
        dummy_neighbors,
        dummy_neighbors,
        dummy_pos_enc,
        dummy_pos_enc,
        dummy_pos_enc,
        dummy_pos_enc,
        dummy_mask,
        dummy_mask,
        1,
        deterministic=True,
    )

    return local_rule_net, params


@pytest.mark.skipif(os.environ.get("RUN_LONG_TESTS", "no") == "no", reason="Skipping long tests")
def test_update_tasknet(
    tasknet: TaskNet, local_rule_net_bundle: tuple[LocalRuleNet, dict], rand_key: jax.random.PRNGKey
):
    local_rule_net, local_rule_net_params = local_rule_net_bundle
    prop_cells_updated = 0.8
    dropout = 0.1
    # import ipdb; ipdb.set_trace()

    param_shapes = jax.tree.map(lambda x: x.shape, tasknet.params)

    new_tasknet = tasknet
    for i in range(3):
        t_start = time.time()
        print("iteration", i)
        new_params, new_states = update_tasknet(
            params=new_tasknet.params,
            hidden_states=new_tasknet.hidden_states,
            positional_encodings=new_tasknet.positional_encodings,
            local_rule_net_apply=local_rule_net.apply,
            local_rule_net_params=local_rule_net_params,
            rand_key=rand_key,
            adj=new_tasknet.adj,
            hidden_dim=new_tasknet.hidden_dim,
            param_names=tuple(new_tasknet.iter_param_names()),
            prop_cells_updated=prop_cells_updated,
            weight_transformer_dropout=dropout,
        )
        new_tasknet = new_tasknet.with_updated(params=new_params, hidden_states=new_states)

        t_end = time.time()
        print(f"done! ({t_end - t_start:0.3f}s)")

    new_param_shapes = jax.tree.map(lambda x: x.shape, new_tasknet.params)

    for layer_name in tasknet.layer_names:
        for param_type in ("kernel", "bias"):
            expected_shape = nested_get(f"{layer_name}.{param_type}", param_shapes)
            shape = nested_get(f"{layer_name}.{param_type}", new_param_shapes)

            assert not jnp.isnan(nested_get(f"{layer_name}.{param_type}", new_tasknet.params)).any()
            assert not jnp.isnan(
                nested_get(f"{layer_name}.{param_type}", new_tasknet.hidden_states)
            ).any()

            assert expected_shape == shape
