from typing import Callable, Sequence

import jax
from frozendict import frozendict
from jax.typing import ArrayLike

from metanca.strategies import (
    register_argument_strategy,
    register_backward_slice_strategy,
    register_forward_slice_strategy,
    register_strategy,
)
from metanca.typing import Prim, SliceInfo

from ._elementwise_nonbarrier_op import (
    arguments_elementwise_nonbarrier,
    forward_neighbors_elementwise_nonbarrier_factory,
    get_forward_slice_info_elementwise_nonbarrier,
)


@register_argument_strategy(strategy_name="add", barrier=False)
def arguments_add(
    focus_param: jax.Array, neighbor_param: jax.Array, primitive: Prim, *args
) -> frozendict:
    return arguments_elementwise_nonbarrier(focus_param, neighbor_param, primitive, *args)


@register_forward_slice_strategy(strategy_name="add")
def get_forward_slice_info_add(
    focus_shape: Sequence[int], neighbor_shape: Sequence[int]
) -> SliceInfo:
    return get_forward_slice_info_elementwise_nonbarrier(focus_shape, neighbor_shape, "add")


@register_strategy("forward_neighbors_factory", "add")
def forward_neighbors_add_factory(
    focus_shape: ArrayLike, neighbor_shape: ArrayLike
) -> Callable[[jax.Array], jax.Array]:
    return forward_neighbors_elementwise_nonbarrier_factory(focus_shape, neighbor_shape, "add")


# ``add`` is elementwise, so the backward slice/factory are identical to the
# forward ones — register the same callables under the backward realm.
@register_backward_slice_strategy(strategy_name="add")
def get_backward_slice_info_add(
    focus_shape: Sequence[int], neighbor_shape: Sequence[int]
) -> SliceInfo:
    return get_forward_slice_info_elementwise_nonbarrier(focus_shape, neighbor_shape, "add")


@register_strategy("backward_neighbors_factory", "add")
def backward_neighbors_add_factory(
    focus_shape: ArrayLike, neighbor_shape: ArrayLike
) -> Callable[[jax.Array], jax.Array]:
    return forward_neighbors_elementwise_nonbarrier_factory(focus_shape, neighbor_shape, "add")
