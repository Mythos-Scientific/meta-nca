from typing import Literal

import jax.lax
import jax.numpy as jnp
import pytest
from frozendict import frozendict

from metanca.strategies import apply_argument_strategy
from metanca.typing import Prim

contexts = ["direct-direct", "direct-indirect", "indirect-direc"]
message_kernels = ["add", "dot_general", "conv_general_dilated"]


@pytest.fixture(params=message_kernels, ids=message_kernels)
def args(request) -> tuple[jax.Array, jax.Array, Prim]:

    match request.param:
        case "add":
            return (
                jnp.zeros((10, 10)),
                jnp.zeros((10, 11)),
                Prim.build(request.param, {}),
                "direct-direct",
            )
        case "dot_general":
            return (
                jnp.zeros((10, 10)),
                jnp.zeros((10, 11)),
                Prim.build(request.param, {"dimension_numbers": (((1,), (0,)), ())}),
                "direct-indirect",
            )
        case "conv_general_dilated":
            focus_param = jnp.zeros((4, 4, 4, 4))
            neighbor_param = jnp.zeros((4, 4, 4, 4))
            return (
                focus_param,
                neighbor_param,
                Prim.build(
                    request.param,
                    {
                        "dimension_numbers": jax.lax.ConvDimensionNumbers(
                            lhs_spec=(0, 3, 1, 2), rhs_spec=(3, 2, 0, 1), out_spec=(0, 3, 1, 2)
                        )
                    },
                ),
                "direct-indirect",
            )


def test_apply_argument_strategy(
    args: tuple[
        jax.Array, jax.Array, Prim, Literal["direct-direct", "direct-indirect", "indirect-direct"]
    ],
):
    focus_param, neighbor_param, primitive, context = args
    args = apply_argument_strategy(primitive.name, focus_param, neighbor_param, primitive, context)
    assert isinstance(args, frozendict)
