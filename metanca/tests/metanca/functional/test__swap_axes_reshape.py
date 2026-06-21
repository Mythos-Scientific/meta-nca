import jax
import jax.numpy as jnp
import pytest

from metanca.functional._swap_axes_reshape import (
    invert_swap_axes_and_reshape,
    swap_axes_and_reshape,
)

KEY = jax.random.key(0)


@pytest.fixture(
    params=[
        {"shape": (10,), "index_dim": 0, "stride": None},
        {"shape": (10, 20), "index_dim": 0, "stride": None},
        {"shape": (10, 20), "index_dim": 1, "stride": None},
        {"shape": (5, 5, 3, 4), "index_dim": 2, "stride": None},
        {"shape": (5, 5, 3, 4), "index_dim": 3, "stride": None},
        {"shape": (5 * 5 * 4, 20), "index_dim": 0, "stride": 4},
    ],
    ids=[
        "bias",
        "matrix::input_dim",
        "matrix::output_dim",
        "filter::input_dim",
        "filter::output_dim",
        "post-filter-matrix::input_dim",
    ],
)
def test_case(request) -> dict:
    global KEY
    KEY, fixture_key = jax.random.split(KEY)
    arr = jax.random.normal(fixture_key, request.param["shape"])
    return {
        "arr": arr,
        "index_dim": request.param["index_dim"],
        "feature_dim_size": 1,
        "stride": request.param["stride"],
    }


def test_swap_axes_reshape(test_case: dict):

    nv = swap_axes_and_reshape(
        test_case["arr"],
        test_case["index_dim"],
        test_case["feature_dim_size"],
        stride=test_case["stride"],
    )
    inverted = invert_swap_axes_and_reshape(
        nv, test_case["arr"].shape, test_case["index_dim"], stride=test_case["stride"]
    )

    assert inverted.shape == test_case["arr"].shape
    assert jnp.allclose(test_case["arr"], inverted)
