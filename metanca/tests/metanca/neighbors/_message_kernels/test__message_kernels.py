import jax
import jax.numpy as jnp
import pytest
from tests.utils import Case, build_all_test_cases, check_assertions

from metanca.functional import batched_slice, swap_axes_and_reshape
from metanca.strategies import apply_strategy
from metanca.typing import SliceInfo

KEY = jax.random.key(0)

TEST_CASE_FILENAME = "slice_info_test_cases.json"


def index_neighbors(
    indices: jax.Array,
    parameter: jax.Array,
    index_dim: int,
    slice_size: int,
    stride: int,
    feature_dim_size: int = 1,
) -> jax.Array:

    neuron_view = swap_axes_and_reshape(parameter, index_dim, feature_dim_size=feature_dim_size)
    sliced_view = batched_slice(indices, neuron_view, slice_size, stride)
    return sliced_view


def slice_neighbors(
    indices: jax.Array,
    parameter: jax.Array,
    slice_info: SliceInfo,
) -> jax.Array:

    indexed_neuron_view = index_neighbors(
        indices,
        parameter,
        index_dim=slice_info["index_dim"],
        slice_size=slice_info["slice_size"],
        stride=slice_info["stride"],
        feature_dim_size=1,
    )
    return indexed_neuron_view


@pytest.fixture(params=build_all_test_cases(TEST_CASE_FILENAME))
def case(request) -> Case:
    global KEY
    case_dict = request.param

    KEY, test_case = Case.create(key=KEY, **case_dict)
    return test_case


def test_apply_strategy_harness(case: Case):
    slice_info = apply_strategy(
        case["realm"],
        case["strategy"],
        case["focus_shape"],
        case["neighbor_shape"],
        **case["factory_kwargs"],
    )
    expected_slice_info = case["slice_info"]
    index_dim = case["slice_info"]["focus"]["index_dim"]

    indices = jnp.asarray(case["indices"]).reshape(1, -1)
    for endpoint in ("focus", "neighbor"):
        parameter = case[endpoint]
        neighbors = slice_neighbors(indices[:, index_dim], parameter, slice_info[endpoint])
        expected_neighbors = slice_neighbors(
            indices[:, index_dim], parameter, expected_slice_info[endpoint]
        )

        err = check_assertions(neighbors, expected_neighbors, prefix=f"[{endpoint}] ", debug=False)
        if err:
            import ipdb

            ipdb.set_trace()
            raise err


def test_conv_general_dilated_forward_bias_to_conv_slice_info():
    slice_info = apply_strategy(
        "forward_slice_info",
        "conv_general_dilated",
        (16,),
        (3, 3, 8, 16),
        rhs_spec=(3, 2, 0, 1),
        context="direct-direct",
    )

    assert slice_info["focus"] == {"index_dim": 0, "stride": 16, "slice_size": 1}
    assert slice_info["neighbor"] == {"index_dim": 3, "stride": 16, "slice_size": 1}


def test_conv_general_dilated_forward_rank3_to_rank4_slice_info():
    slice_info = apply_strategy(
        "forward_slice_info",
        "conv_general_dilated",
        (3, 3, 16),
        (1, 1, 16, 32),
        rhs_spec=(3, 2, 0, 1),
        context="direct-indirect",
    )

    assert slice_info["focus"] == {"index_dim": 2, "stride": 16, "slice_size": 1}
    assert slice_info["neighbor"] == {"index_dim": 2, "stride": 16, "slice_size": 1}


def test_conv_general_dilated_forward_same_rank_direct_indirect_aligns_output_channels():
    slice_info = apply_strategy(
        "forward_slice_info",
        "conv_general_dilated",
        (3, 3, 16, 16),
        (3, 3, 3, 16),
        rhs_spec=(3, 2, 0, 1),
        context="direct-indirect",
    )

    assert slice_info["focus"] == {"index_dim": 3, "stride": 16, "slice_size": 1}
    assert slice_info["neighbor"] == {"index_dim": 3, "stride": 16, "slice_size": 1}


def test_conv_general_dilated_forward_same_rank_direct_direct_prefers_matching_channels():
    slice_info = apply_strategy(
        "forward_slice_info",
        "conv_general_dilated",
        (3, 3, 16, 32),
        (1, 1, 16, 32),
        rhs_spec=(3, 2, 0, 1),
        context="direct-direct",
    )

    assert slice_info["focus"] == {"index_dim": 3, "stride": 32, "slice_size": 1}
    assert slice_info["neighbor"] == {"index_dim": 3, "stride": 32, "slice_size": 1}
