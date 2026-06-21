from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from frozendict import frozendict
from jax.typing import ArrayLike

from metanca.typing import MsgSliceInfo, Prim, SliceInfo


def _elementwise_nonbarrier_op(
    focus_shape: Sequence[int], neighbor_shape: Sequence[int], op_name: str
) -> tuple[int, int]:
    focus_rank, neighbor_rank = map(len, (focus_shape, neighbor_shape))
    if focus_rank == neighbor_rank and focus_shape == neighbor_shape:
        return focus_rank, neighbor_rank
    if neighbor_rank == 1 and focus_rank != 1 and focus_shape[-1] == neighbor_shape[0]:
        return focus_rank, neighbor_rank
    raise TypeError(
        f"Shape mismatch: {op_name} with {neighbor_shape} cannot be broadcast onto {focus_shape}"
    )


def arguments_elementwise_nonbarrier(
    focus_param: jax.Array, neighbor_param: jax.Array, primitive: Prim, *args
) -> frozendict:
    return frozendict(
        {
            "focus_shape": focus_param.shape,
            "neighbor_shape": neighbor_param.shape,
        }
    )


def get_forward_slice_info_elementwise_nonbarrier(
    focus_shape: Sequence[int], neighbor_shape: Sequence[int], op_name: str
) -> MsgSliceInfo:
    focus_rank, neighbor_rank = _elementwise_nonbarrier_op(focus_shape, neighbor_shape, op_name)
    if focus_rank == neighbor_rank:
        return MsgSliceInfo(
            focus=SliceInfo(index_dim=0, stride=1, slice_size=1),
            neighbor=SliceInfo(index_dim=0, stride=1, slice_size=1),
        )
    return MsgSliceInfo(
        focus=SliceInfo(index_dim=focus_rank - 1, stride=focus_shape[-1], slice_size=1),
        neighbor=SliceInfo(index_dim=0, stride=neighbor_shape[0], slice_size=1),
    )


def forward_neighbors_elementwise_nonbarrier_factory(
    focus_shape: ArrayLike, neighbor_shape: ArrayLike, op_name: str
) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    focus_rank, neighbor_rank = _elementwise_nonbarrier_op(focus_shape, neighbor_shape, op_name)
    if focus_rank == neighbor_rank:

        def forward_neighbors_equal_rank(indices: jax.Array) -> tuple[jax.Array, jax.Array]:
            self_slice = neighbor_slice = jnp.stack([indices, indices + 1], axis=1)
            return self_slice, neighbor_slice

        return forward_neighbors_equal_rank

    range_select_leading_dims = (
        jnp.zeros((focus_rank - 1, 2), dtype=int)
        .at[:, 1]
        .set(jnp.asarray(focus_shape, dtype=int)[:-1])
    )

    def forward_neighbors_broadcast_vector(
        indices: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        self_slice = (
            jnp.stack([indices, indices + 1], axis=1).at[:-1, :].set(range_select_leading_dims)
        )
        neighbor_slice = jnp.stack([indices[-1], indices[-1] + 1]).reshape(1, 2)
        return self_slice, neighbor_slice

    return forward_neighbors_broadcast_vector
