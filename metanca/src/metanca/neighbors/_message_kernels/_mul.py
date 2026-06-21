from typing import Callable, Sequence

import jax
from frozendict import frozendict
from jax.typing import ArrayLike

from metanca.strategies import (
    register_argument_strategy,
    register_forward_slice_strategy,
    register_strategy,
)
from metanca.typing import Prim, SliceInfo

from ._elementwise_nonbarrier_op import (
    arguments_elementwise_nonbarrier,
    forward_neighbors_elementwise_nonbarrier_factory,
    get_forward_slice_info_elementwise_nonbarrier,
)


@register_argument_strategy(strategy_name="mul", barrier=False)
def arguments_mul(
    focus_param: jax.Array, neighbor_param: jax.Array, primitive: Prim, *args
) -> frozendict:
    return arguments_elementwise_nonbarrier(focus_param, neighbor_param, primitive, *args)


@register_forward_slice_strategy(strategy_name="mul")
def get_forward_slice_info_mul(
    focus_shape: Sequence[int], neighbor_shape: Sequence[int]
) -> SliceInfo:
    return get_forward_slice_info_elementwise_nonbarrier(focus_shape, neighbor_shape, "mul")


@register_strategy("forward_neighbors_factory", "mul")
def forward_neighbors_mul_factory(
    focus_shape: ArrayLike, neighbor_shape: ArrayLike
) -> Callable[[jax.Array], jax.Array]:
    return forward_neighbors_elementwise_nonbarrier_factory(focus_shape, neighbor_shape, "mul")
