from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from frozendict import frozendict
from jax.typing import ArrayLike

from metanca.strategies import (
    register_argument_strategy,
    register_backward_slice_strategy,
    register_forward_slice_strategy,
    register_strategy,
)
from metanca.typing import MsgSliceInfo, Prim, PrimitiveContext, SliceInfo


@register_argument_strategy(strategy_name="conv_general_dilated", barrier=True)
def arguments_conv_general_dilated(
    focus_param: jax.Array, neighbor_param: jax.Array, primitive: Prim, context: PrimitiveContext
) -> frozendict:
    return frozendict(
        {
            "focus_shape": focus_param.shape,
            "neighbor_shape": neighbor_param.shape,
            "rhs_spec": primitive.params["dimension_numbers"].rhs_spec,
            "context": context,
        }
    )


@register_forward_slice_strategy(strategy_name="conv_general_dilated")
def get_forward_slice_info_conv_general_dilated(
    focus_shape: Sequence[int],
    neighbor_shape: Sequence[int],
    rhs_spec: Sequence[int],
    context: PrimitiveContext,
) -> MsgSliceInfo:

    focus_rank = len(focus_shape)
    neighbor_rank = len(neighbor_shape)

    output_filter_dim, input_filter_dim, *_ = rhs_spec

    if neighbor_rank == focus_rank:
        focus_out = focus_shape[output_filter_dim]
        neighbor_in = neighbor_shape[input_filter_dim]
        neighbor_out = neighbor_shape[output_filter_dim]
        if neighbor_in == focus_out and neighbor_out != focus_out:
            aligned_dim = input_filter_dim
        elif neighbor_out == focus_out:
            aligned_dim = output_filter_dim
        else:
            aligned_dim = input_filter_dim if context == "direct-direct" else output_filter_dim
        # conv to a conv
        return MsgSliceInfo(
            focus=SliceInfo(
                index_dim=output_filter_dim, stride=focus_shape[output_filter_dim], slice_size=1
            ),
            neighbor=SliceInfo(
                index_dim=aligned_dim, stride=neighbor_shape[aligned_dim], slice_size=1
            ),
        )

    elif focus_rank == 1:
        match context:
            case "direct-direct":
                aligned_dim = output_filter_dim
            case "direct-indirect":
                aligned_dim = input_filter_dim
            case _:
                raise ValueError(f"{context=} not supported.")

        return MsgSliceInfo(
            focus=SliceInfo(index_dim=0, stride=focus_shape[0], slice_size=1),
            neighbor=SliceInfo(
                index_dim=aligned_dim, stride=neighbor_shape[aligned_dim], slice_size=1
            ),
        )

    elif focus_rank + 2 == neighbor_rank:
        # weight to a filter
        return MsgSliceInfo(
            focus=SliceInfo(index_dim=1, stride=focus_shape[1], slice_size=1),
            neighbor=SliceInfo(index_dim=input_filter_dim, stride=focus_shape[1], slice_size=1),
        )

    elif focus_rank + 1 == neighbor_rank:
        # reduced-rank filter (e.g., depthwise-like) to a full filter
        aligned_dim = focus_rank - 1
        return MsgSliceInfo(
            focus=SliceInfo(index_dim=aligned_dim, stride=focus_shape[aligned_dim], slice_size=1),
            neighbor=SliceInfo(
                index_dim=input_filter_dim,
                stride=focus_shape[aligned_dim],
                slice_size=1,
            ),
        )


@register_backward_slice_strategy(strategy_name="conv_general_dilated")
def get_backward_slice_info_conv_general_dilated(
    focus_shape: Sequence[int],
    neighbor_shape: Sequence[int],
    rhs_spec: Sequence[int],
    context: PrimitiveContext,
) -> SliceInfo:

    focus_rank = len(focus_shape)
    neighbor_rank = len(neighbor_shape)

    output_filter_dim, input_filter_dim, *_ = rhs_spec

    if focus_rank == neighbor_rank:

        return MsgSliceInfo(
            focus=SliceInfo(
                index_dim=input_filter_dim, stride=focus_shape[input_filter_dim], slice_size=1
            ),
            neighbor=SliceInfo(
                index_dim=output_filter_dim, stride=neighbor_shape[output_filter_dim], slice_size=1
            ),
        )

    elif focus_rank + 2 == neighbor_rank:
        return MsgSliceInfo(
            focus=SliceInfo(
                index_dim=0,
                stride=neighbor_shape[output_filter_dim],
                slice_size=focus_shape[0],
            ),
            neighbor=SliceInfo(
                index_dim=output_filter_dim, stride=neighbor_shape[output_filter_dim], slice_size=1
            ),
        )

    elif neighbor_rank == 1 and focus_rank != len(rhs_spec):

        # flatten(maxpool(conv(x, W1) + b1)) dot W2 -- focus=W2; neighbor=b1
        return MsgSliceInfo(
            focus=SliceInfo(index_dim=0, stride=neighbor_shape[0], slice_size=focus_shape[0]),
            neighbor=SliceInfo(index_dim=0, stride=neighbor_shape[0], slice_size=1),
        )

    elif neighbor_rank == 1 and focus_rank == len(rhs_spec):
        # conv(conv(x, W1) + b1, W2) -- focus=W2; neighbor=b1
        return MsgSliceInfo(
            focus=SliceInfo(
                index_dim=input_filter_dim, stride=focus_shape[input_filter_dim], slice_size=1
            ),
            neighbor=SliceInfo(index_dim=0, stride=neighbor_shape[0], slice_size=1),
        )


@register_strategy("forward_neighbors_factory", "conv_general_dilated")
def forward_neighbors_conv_general_dilated_factory(
    focus_shape: ArrayLike,
    neighbor_shape: ArrayLike,
    rhs_spec: tuple[int, ...],
    context: PrimitiveContext,
) -> Callable[[jax.Array], jax.Array]:

    focus_rank = len(focus_shape)
    neighbor_rank = len(neighbor_shape)
    output_filter_dim, input_filter_dim, *_ = rhs_spec

    if focus_rank == 1:

        match context:
            case "direct-direct":
                aligned_dim = output_filter_dim
            case "direct-indirect":
                aligned_dim = input_filter_dim
            case _:
                raise ValueError(f"{context=} not supported.")

        if focus_shape != neighbor_shape[aligned_dim]:
            raise RuntimeError(f"{focus_shape=}; {neighbor_shape[aligned_dim]=}")

        def forward_neighbors_conv_general_dilated(
            indices: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            self_slice = jnp.asarray([0, indices[0], focus_shape[0], focus_shape[0]], dtype=int)
            neighbor_slice = jnp.asarray(
                [aligned_dim, indices[0], neighbor_shape[aligned_dim], neighbor_shape[aligned_dim]],
                dtype=int,
            )
            return self_slice, neighbor_slice

    elif neighbor_rank == focus_rank + 2:

        def forward_neighbors_conv_general_dilated(
            indices: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            self_slice = jnp.asarray([1, indices[1], focus_shape[1], focus_shape[1]], dtype=int)
            neighbor_slice = jnp.asarray(
                [
                    input_filter_dim,
                    indices[1],
                    neighbor_shape[input_filter_dim],
                    neighbor_shape[input_filter_dim],
                ],
                dtype=int,
            )
            return self_slice, neighbor_slice

    elif neighbor_rank == focus_rank + 1:
        aligned_dim = focus_rank - 1

        def forward_neighbors_conv_general_dilated(
            indices: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            self_slice = jnp.asarray(
                [
                    aligned_dim,
                    indices[aligned_dim],
                    focus_shape[aligned_dim],
                    focus_shape[aligned_dim],
                ],
                dtype=int,
            )
            neighbor_slice = jnp.asarray(
                [
                    input_filter_dim,
                    indices[aligned_dim],
                    neighbor_shape[input_filter_dim],
                    neighbor_shape[input_filter_dim],
                ],
                dtype=int,
            )
            return self_slice, neighbor_slice

    elif neighbor_rank == focus_rank:
        focus_out = focus_shape[output_filter_dim]
        neighbor_in = neighbor_shape[input_filter_dim]
        neighbor_out = neighbor_shape[output_filter_dim]
        if neighbor_in == focus_out and neighbor_out != focus_out:
            aligned_dim = input_filter_dim
        elif neighbor_out == focus_out:
            aligned_dim = output_filter_dim
        else:
            aligned_dim = input_filter_dim if context == "direct-direct" else output_filter_dim

        def forward_neighbors_conv_general_dilated(
            indices: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            self_slice = jnp.asarray(
                [
                    output_filter_dim,
                    indices[output_filter_dim],
                    focus_shape[output_filter_dim],
                    focus_shape[output_filter_dim],
                ],
                dtype=int,
            )
            neighbor_slice = jnp.asarray(
                [
                    aligned_dim,
                    indices[output_filter_dim],
                    neighbor_shape[aligned_dim],
                    neighbor_shape[aligned_dim],
                ],
                dtype=int,
            )
            return self_slice, neighbor_slice

    return forward_neighbors_conv_general_dilated


@register_strategy("backward_neighbors_factory", "conv_general_dilated")
def backward_neighbors_conv_general_dilated_factory(
    focus_shape: ArrayLike,
    neighbor_shape: ArrayLike,
    rhs_spec: tuple[int, ...],
    context: PrimitiveContext,
) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:

    neighbor_rank = len(neighbor_shape)
    focus_rank = len(focus_shape)
    output_filter_dim, input_filter_dim, *_ = rhs_spec

    if focus_rank == neighbor_rank + 2:
        raise NotImplementedError("cases likes conv(xW1 + b1, W2) -- focus=W2; neighbor=W1")

    elif neighbor_rank == focus_rank + 2:

        def backward_neighbors_conv_general_dilated(
            indices: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:

            # axis, start index, limit end, stride
            self_slice = jnp.asarray(
                [0, indices[0], focus_shape[0], neighbor_shape[output_filter_dim]], dtype=int
            )
            neighbor_slice = jnp.asarray(
                [
                    output_filter_dim,
                    indices[0],
                    neighbor_shape[output_filter_dim],
                    neighbor_shape[output_filter_dim],
                ]
            )
            return self_slice, neighbor_slice

    elif focus_rank == neighbor_rank:

        # (axis, start, end, stride)
        def backward_neighbors_conv_general_dilated(
            indices: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            # (axis, start, limit, stride)
            self_slice = jnp.asarray(
                [
                    input_filter_dim,
                    indices[input_filter_dim],
                    focus_shape[input_filter_dim],
                    focus_shape[input_filter_dim],
                ],
                dtype=int,
            )
            neighbor_slice = jnp.asarray(
                [
                    output_filter_dim,
                    indices[input_filter_dim],
                    neighbor_shape[output_filter_dim],
                    neighbor_shape[output_filter_dim],
                ],
                dtype=int,
            )
            return self_slice, neighbor_slice

    elif neighbor_rank == 1:

        if len(focus_shape) != len(rhs_spec):
            # flatten(maxpool(conv(x, W1) + b1)) dot W2 -- focus=W2; neighbor=b1
            def backward_neighbors_conv_general_dilated(
                indices: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
                self_slice = jnp.asarray(
                    [0, indices[0], focus_shape[0], neighbor_shape[0]], dtype=int
                )
                neighbor_slice = jnp.asarray(
                    [0, indices[0], neighbor_shape[0], neighbor_shape[0]], dtype=int
                )
                return self_slice, neighbor_slice

        else:
            # conv(conv(x, W1) + b1, W2) -- focus=W; neighbor=b1
            def backward_neighbors_conv_general_dilated(
                indices: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
                self_slice = jnp.asarray(
                    [
                        input_filter_dim,
                        indices[input_filter_dim],
                        focus_shape[input_filter_dim],
                        focus_shape[input_filter_dim],
                    ],
                    dtype=int,
                )
                neighbor_slice = jnp.asarray(
                    [0, indices[input_filter_dim], neighbor_shape[0], neighbor_shape[0]], dtype=int
                )
                return self_slice, neighbor_slice

    return backward_neighbors_conv_general_dilated
