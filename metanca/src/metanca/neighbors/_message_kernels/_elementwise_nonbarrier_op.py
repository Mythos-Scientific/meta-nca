from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from frozendict import frozendict
from jax.typing import ArrayLike

from metanca.typing import MsgSliceInfo, Prim, SliceInfo


def _classify_elementwise(
    focus_shape: Sequence[int], neighbor_shape: Sequence[int], op_name: str
) -> str:
    """Classify how a non-barrier elementwise op pairs the two parameter shapes.

    Returns one of:
        - ``"equal_shape"``: focus and neighbor are pointwise paired (e.g. two
          parameters of identical shape sharing the same op).
        - ``"broadcast_vector"``: a 1-D neighbor (e.g. bias) broadcasts onto the
          focus's last axis (e.g. Dense kernel + bias).
        - ``"shared_trailing_channel"``: both ranks >= 2 and the trailing axis
          has the same size — typical of two transformer-style weights writing
          into the same residual channel (e.g. ``embedding[V,D]`` and
          ``out_proj[D,D]`` both write to channel D).
    """
    focus_rank, neighbor_rank = map(len, (focus_shape, neighbor_shape))
    if focus_rank == neighbor_rank and focus_shape == neighbor_shape:
        return "equal_shape"
    if neighbor_rank == 1 and focus_rank != 1 and focus_shape[-1] == neighbor_shape[0]:
        return "broadcast_vector"
    if focus_rank >= 2 and neighbor_rank >= 2 and focus_shape[-1] == neighbor_shape[-1]:
        return "shared_trailing_channel"
    raise TypeError(
        f"Shape mismatch: {op_name} with {neighbor_shape} cannot be paired with {focus_shape}"
    )


def _elementwise_nonbarrier_op(
    focus_shape: Sequence[int], neighbor_shape: Sequence[int], op_name: str
) -> tuple[int, int]:
    """Backwards-compatible shape-validation wrapper.

    Raises ``TypeError`` if the shapes can't be paired by any supported case.
    """
    _classify_elementwise(focus_shape, neighbor_shape, op_name)
    return len(focus_shape), len(neighbor_shape)


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
    case = _classify_elementwise(focus_shape, neighbor_shape, op_name)
    focus_rank, neighbor_rank = len(focus_shape), len(neighbor_shape)

    if case == "equal_shape":
        return MsgSliceInfo(
            focus=SliceInfo(index_dim=0, stride=1, slice_size=1),
            neighbor=SliceInfo(index_dim=0, stride=1, slice_size=1),
        )
    if case == "broadcast_vector":
        return MsgSliceInfo(
            focus=SliceInfo(index_dim=focus_rank - 1, stride=focus_shape[-1], slice_size=1),
            neighbor=SliceInfo(index_dim=0, stride=neighbor_shape[0], slice_size=1),
        )
    # shared_trailing_channel: both ranks >= 2 and the last axes match.
    return MsgSliceInfo(
        focus=SliceInfo(index_dim=focus_rank - 1, stride=focus_shape[-1], slice_size=1),
        neighbor=SliceInfo(index_dim=neighbor_rank - 1, stride=neighbor_shape[-1], slice_size=1),
    )


def forward_neighbors_elementwise_nonbarrier_factory(
    focus_shape: ArrayLike, neighbor_shape: ArrayLike, op_name: str
) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    case = _classify_elementwise(focus_shape, neighbor_shape, op_name)
    focus_rank, neighbor_rank = len(focus_shape), len(neighbor_shape)

    if case == "equal_shape":

        def forward_neighbors_equal_rank(indices: jax.Array) -> tuple[jax.Array, jax.Array]:
            self_slice = neighbor_slice = jnp.stack([indices, indices + 1], axis=1)
            return self_slice, neighbor_slice

        return forward_neighbors_equal_rank

    focus_range_leading = (
        jnp.zeros((focus_rank - 1, 2), dtype=int)
        .at[:, 1]
        .set(jnp.asarray(focus_shape, dtype=int)[:-1])
    )

    if case == "broadcast_vector":

        def forward_neighbors_broadcast_vector(
            indices: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            self_slice = (
                jnp.stack([indices, indices + 1], axis=1).at[:-1, :].set(focus_range_leading)
            )
            neighbor_slice = jnp.stack([indices[-1], indices[-1] + 1]).reshape(1, 2)
            return self_slice, neighbor_slice

        return forward_neighbors_broadcast_vector

    # shared_trailing_channel: pair focus output channel j with neighbor output
    # channel j; both take a 1-element slice along their respective last axes.
    neighbor_range_leading = (
        jnp.zeros((neighbor_rank - 1, 2), dtype=int)
        .at[:, 1]
        .set(jnp.asarray(neighbor_shape, dtype=int)[:-1])
    )

    def forward_neighbors_shared_trailing_channel(
        indices: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        self_slice = jnp.stack([indices, indices + 1], axis=1).at[:-1, :].set(focus_range_leading)
        neighbor_slice_last = jnp.stack([indices[-1], indices[-1] + 1]).reshape(1, 2)
        neighbor_slice = jnp.concatenate([neighbor_range_leading, neighbor_slice_last], axis=0)
        return self_slice, neighbor_slice

    return forward_neighbors_shared_trailing_channel
