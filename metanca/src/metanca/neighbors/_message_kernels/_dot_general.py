import operator
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


def annotate_dot_general_contraction_axes(
    shape: jax.Array, contraction_axes: jax.Array
) -> ArrayLike:

    if len(shape) < len(contraction_axes):
        raise TypeError("Shape should be longer than contraction axes.")
    elif any(ax >= len(shape) for ax in contraction_axes):
        raise TypeError("Contraction axes out of range.")

    mask = [False] * len(shape)
    for ax in contraction_axes:
        if mask[ax]:
            raise TypeError(f"Repeated contraction axes in {contraction_axes}")
        mask[ax] = True

    return jnp.asarray(mask, dtype=jnp.bool)


@register_argument_strategy(strategy_name="dot_general", barrier=True)
def arguments_dot_general(
    focus_param: jax.Array,
    neighbor_param: jax.Array,
    primitive: Prim,
    context: PrimitiveContext,
) -> frozendict:
    (lhs_ctrx, rhs_ctrx), _ = primitive.params["dimension_numbers"]

    return frozendict(
        {
            "focus_shape": focus_param.shape,
            "neighbor_shape": neighbor_param.shape,
            "contraction_axes": tuple(zip(lhs_ctrx, rhs_ctrx)),
            "context": context,
        }
    )


unpack_dot_general_info = operator.itemgetter(
    "focus_shape", "neighbor_shape", "contraction_axes", "context"
)


@register_forward_slice_strategy(strategy_name="dot_general")
def get_forward_slice_info_dot_general(
    focus_shape: Sequence[int],
    neighbor_shape: Sequence[int],
    contraction_axes: Sequence[tuple[int, int]],
    context: PrimitiveContext,
) -> SliceInfo:
    lhs_ctrx, rhs_ctrx = zip(*contraction_axes)
    neighbor_rank, focus_rank = map(len, (neighbor_shape, focus_shape))

    if neighbor_rank == focus_rank:
        return MsgSliceInfo(
            focus=SliceInfo(index_dim=lhs_ctrx[0], stride=focus_shape[lhs_ctrx[0]], slice_size=1),
            neighbor=SliceInfo(
                index_dim=rhs_ctrx[0], stride=neighbor_shape[rhs_ctrx[0]], slice_size=1
            ),
        )

    elif focus_rank > neighbor_rank:
        return MsgSliceInfo(
            focus=SliceInfo(
                index_dim=focus_rank - 1, stride=focus_shape[focus_rank - 1], slice_size=1
            ),
            neighbor=SliceInfo(
                index_dim=rhs_ctrx[0],
                stride=focus_shape[focus_rank - 1],
                slice_size=neighbor_shape[rhs_ctrx[0]],
            ),
        )

    elif focus_rank == 1:
        match context:
            case "direct-direct":
                return MsgSliceInfo(
                    focus=SliceInfo(index_dim=0, stride=focus_shape[0], slice_size=1),
                    neighbor=SliceInfo(
                        index_dim=lhs_ctrx[0], stride=neighbor_shape[lhs_ctrx[0]], slice_size=1
                    ),
                )

            case "direct-indirect":
                # find out whether we are sitting on a filter-bias
                # or a normal bias.
                if focus_shape[0] != neighbor_shape[rhs_ctrx[0]]:
                    # filter-bias to a weight
                    return MsgSliceInfo(
                        focus=SliceInfo(index_dim=0, stride=focus_shape[0], slice_size=1),
                        neighbor=SliceInfo(
                            index_dim=rhs_ctrx[0],
                            stride=focus_shape[0],
                            slice_size=neighbor_shape[rhs_ctrx[0]],
                        ),
                    )
                else:
                    return MsgSliceInfo(
                        focus=SliceInfo(index_dim=0, stride=1, slice_size=1),
                        neighbor=SliceInfo(index_dim=rhs_ctrx[0], stride=1, slice_size=1),
                    )


@register_backward_slice_strategy(strategy_name="dot_general")
def get_backward_slice_info_dot_general(
    focus_shape: Sequence[int],
    neighbor_shape: Sequence[int],
    contraction_axes: Sequence[tuple[int, int]],
    context: PrimitiveContext,
) -> SliceInfo:
    lhs_ctrx, rhs_ctrx = zip(*contraction_axes)
    neighbor_rank, focus_rank = map(len, (neighbor_shape, focus_shape))

    if neighbor_rank == focus_rank:
        return MsgSliceInfo(
            focus=SliceInfo(index_dim=rhs_ctrx[0], stride=focus_shape[rhs_ctrx[0]], slice_size=1),
            neighbor=SliceInfo(
                index_dim=lhs_ctrx[0], stride=neighbor_shape[lhs_ctrx[0]], slice_size=1
            ),
        )
    elif neighbor_rank > focus_rank:
        return MsgSliceInfo(
            focus=SliceInfo(
                index_dim=rhs_ctrx[0],
                stride=neighbor_shape[neighbor_rank - 1],
                slice_size=focus_shape[rhs_ctrx[0]],
            ),
            neighbor=SliceInfo(
                index_dim=neighbor_rank - 1, stride=neighbor_shape[neighbor_rank - 1], slice_size=1
            ),
        )
    elif neighbor_rank == 1:
        if neighbor_shape[0] != focus_shape[rhs_ctrx[0]]:
            # conv bias to a flattened weight
            return MsgSliceInfo(
                focus=SliceInfo(
                    index_dim=rhs_ctrx[0],
                    stride=neighbor_shape[0],
                    slice_size=focus_shape[rhs_ctrx[0]],
                ),
                neighbor=SliceInfo(index_dim=0, stride=neighbor_shape[0], slice_size=1),
            )
        else:
            return MsgSliceInfo(
                focus=SliceInfo(
                    index_dim=rhs_ctrx[0], stride=focus_shape[rhs_ctrx[0]], slice_size=1
                ),
                neighbor=SliceInfo(index_dim=0, stride=neighbor_shape[0], slice_size=1),
            )


@register_strategy("backward_neighbors_factory", "dot_general")
def backward_neighbors_dot_general_factory(
    focus_shape: jax.Array,
    neighbor_shape: jax.Array,
    contraction_axes: jax.Array,
    context: PrimitiveContext,
) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:

    focus_rank = len(focus_shape)
    neighbor_rank = len(neighbor_shape)

    if neighbor_rank == focus_rank:
        # weight is focus, neighbor is weight
        annotated_axes_neighbor = jnp.isin(jnp.arange(neighbor_rank), contraction_axes[:, 0])
        annotated_axes_focus = jnp.isin(jnp.arange(focus_rank), contraction_axes[:, 1])

        range_select_neighbor = jnp.asarray(neighbor_shape, dtype=int)[~annotated_axes_neighbor]
        range_select_focus = jnp.asarray(focus_shape, dtype=int)[~annotated_axes_focus]

        neighbor_slice_template = (
            jnp.zeros((neighbor_rank, 2), dtype=int)
            .at[~annotated_axes_neighbor, 1]
            .set(range_select_neighbor)
        )

        self_slice_template = (
            jnp.zeros((focus_rank, 2), dtype=int)
            .at[~annotated_axes_focus, 1]
            .set(range_select_focus)
        )

        def backward_neighbors_dot_general(indices: jax.Array) -> tuple[jax.Array, jax.Array]:
            index_select = jnp.stack(
                [indices[annotated_axes_focus], indices[annotated_axes_focus] + 1], axis=1
            )
            neighbor_slice = neighbor_slice_template.at[annotated_axes_neighbor, :].set(
                index_select
            )
            self_slice = self_slice_template.at[annotated_axes_focus, :].set(index_select)
            return self_slice, neighbor_slice

    elif neighbor_rank == contraction_axes.shape[0]:
        # import ipdb; ipdb.set_trace()
        # cases like (xW1 + b1)W2
        focus_shape_arr = jnp.asarray(focus_shape)
        contracting_dim_shapes = focus_shape_arr[contraction_axes[:, 1]]
        annotated_axes_focus = jnp.isin(jnp.arange(focus_rank), contraction_axes[:, 1])
        if contracting_dim_shapes[-1] == neighbor_shape:
            range_select_focus = focus_shape_arr[~annotated_axes_focus]
            self_slice_template = (
                jnp.zeros((focus_rank, 2), dtype=int)
                .at[~annotated_axes_focus, 1]
                .set(range_select_focus)
            )

            def backward_neighbors_dot_general(
                indices: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:

                index_select = jnp.stack(
                    [indices[annotated_axes_focus], indices[annotated_axes_focus] + 1], axis=1
                )
                neighbor_slice = index_select.reshape(1, 2)
                self_slice = self_slice_template.at[annotated_axes_focus, :].set(index_select)
                return self_slice, neighbor_slice

        else:
            # cases like flatten(conv(x, W1) + b1) W2
            def backward_neighbors_dot_general(
                indices: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:

                neighbor_slice = jnp.asarray(
                    [0, indices[contraction_axes[0, 1]], neighbor_shape[0], neighbor_shape[0]],
                    dtype=int,
                )
                self_slice = jnp.asarray(
                    [
                        contraction_axes[0, 1],
                        indices[contraction_axes[0, 1]],
                        focus_shape[contraction_axes[0, 1]],
                        neighbor_shape[0],
                    ],
                    dtype=int,
                )
                return self_slice, neighbor_slice

    elif neighbor_rank > focus_rank:
        # filter is focus, neighbor is weight
        # flatten(maxpool(conv(x, W1)))W2 -- W1 is the focus and W2 is the neighbor
        annotated_axes_focus = jnp.isin(jnp.arange(focus_rank), contraction_axes[:, 1])
        range_select_focus = jnp.asarray(focus_shape, dtype=int)[~annotated_axes_focus]
        output_filter_dim = 3

        def backward_neighbors_dot_general(indices: jax.Array) -> tuple[jax.Array, jax.Array]:
            neighbor_slice = jnp.asarray(
                [
                    output_filter_dim,
                    indices[contraction_axes[0, 1]],
                    neighbor_shape[output_filter_dim],
                    neighbor_shape[output_filter_dim],
                ],
                dtype=int,
            )

            self_slice = jnp.asarray(
                [
                    contraction_axes[0, 1],
                    indices[contraction_axes[0, 1]],
                    focus_shape[contraction_axes[0, 1]],
                    neighbor_shape[output_filter_dim],
                ],
                dtype=int,
            )
            return self_slice, neighbor_slice

    return backward_neighbors_dot_general


@register_strategy("forward_neighbors_factory", "dot_general")
def forward_neighbors_dot_general_factory(
    focus_shape: jax.Array,
    neighbor_shape: jax.Array,
    contraction_axes: jax.Array,
    context: PrimitiveContext,
) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:

    focus_rank = len(focus_shape)
    neighbor_rank = len(neighbor_shape)

    if focus_rank == neighbor_rank:
        # weight is focus, neighbor is weight
        annotated_axes_focus = jnp.isin(jnp.arange(focus_rank), contraction_axes[:, 0])
        annotated_axes_neighbor = jnp.isin(jnp.arange(neighbor_rank), contraction_axes[:, 1])

        range_select_focus = jnp.asarray(focus_shape, dtype=int)[~annotated_axes_focus]
        range_select_neighbor = jnp.asarray(neighbor_shape, dtype=int)[~annotated_axes_neighbor]

        self_slice_template = (
            jnp.zeros((focus_rank, 2), dtype=int)
            .at[~annotated_axes_focus, 1]
            .set(range_select_focus)
        )

        neighbor_slice_template = (
            jnp.zeros((neighbor_rank, 2), dtype=int)
            .at[~annotated_axes_neighbor, 1]
            .set(range_select_neighbor)
        )

        def forward_neighbors_dot_general(indices: jax.Array) -> tuple[jax.Array, jax.Array]:
            index_select = jnp.stack(
                [indices[annotated_axes_focus], indices[annotated_axes_focus] + 1], axis=1
            )
            self_slice = self_slice_template.at[annotated_axes_focus, :].set(index_select)
            neighbor_slice = neighbor_slice_template.at[annotated_axes_neighbor, :].set(
                index_select
            )
            return self_slice, neighbor_slice

    elif focus_rank == contraction_axes.shape[0]:
        if context == "direct-direct":
            # (xW1 + b1) b1 is focus, W1 is the neighbor

            annotated_axes_neighbor_mask = jnp.isin(
                jnp.arange(neighbor_rank), contraction_axes[:, 1]
            )
            annotated_axes_neighbor = jnp.arange(neighbor_rank, dtype=int)[
                annotated_axes_neighbor_mask
            ]
            jnp.arange(neighbor_rank, dtype=int)[~annotated_axes_neighbor_mask]

            range_select_neighbor = jnp.asarray(neighbor_shape, dtype=int)[annotated_axes_neighbor]

            neighbor_slice_template = (
                jnp.zeros((neighbor_rank, 2), dtype=int)
                .at[annotated_axes_neighbor, 1]
                .set(range_select_neighbor)
            )

            def forward_neighbors_dot_general(
                indices: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:

                index_select = jnp.stack([indices, indices + 1], axis=1)
                self_slice = index_select
                neighbor_slice = neighbor_slice_template.at[annotated_axes_neighbor, :].set(
                    index_select
                )
                return self_slice, neighbor_slice

        elif context == "direct-indirect":
            # cases like flatten(conv(x, W1) + b1) W2 and (xW1 + b1)W2
            neighbor_shape_arr = jnp.asarray(neighbor_shape)
            contracting_dim_shapes = neighbor_shape_arr[contraction_axes[:, 1]]
            if contracting_dim_shapes[-1] == focus_shape:
                annotated_axes_neighbor = jnp.isin(
                    jnp.arange(neighbor_rank), contraction_axes[:, 1]
                )
                range_select_neighbor = jnp.asarray(neighbor_shape, dtype=int)[
                    ~annotated_axes_neighbor
                ]

                neighbor_slice_template = (
                    jnp.zeros((neighbor_rank, 2), dtype=int)
                    .at[~annotated_axes_neighbor, 1]
                    .set(range_select_neighbor)
                )

                def forward_neighbors_dot_general(
                    indices: jax.Array,
                ) -> tuple[jax.Array, jax.Array]:

                    index_select = jnp.stack([indices, indices + 1], axis=1)
                    self_slice = index_select
                    neighbor_slice = neighbor_slice_template.at[annotated_axes_neighbor, :].set(
                        index_select
                    )
                    return self_slice, neighbor_slice

            else:

                def forward_neighbors_dot_general(
                    indices: jax.Array,
                ) -> tuple[jax.Array, jax.Array]:
                    self_slice = jnp.asarray(
                        [0, indices[0], focus_shape[0], focus_shape[0]], dtype=int
                    )
                    neighbor_slice = jnp.asarray(
                        [
                            contraction_axes[0, 1],
                            indices[0],
                            neighbor_shape[contraction_axes[0, 1]],
                            focus_shape[0],
                        ],
                        dtype=int,
                    )
                    return self_slice, neighbor_slice

        else:
            raise ValueError(f"Unsupported context: {context}")

    elif focus_rank > neighbor_rank:
        # filter is focus, neighbor is weight
        # flatten(maxpool(conv(x, W1)))W2 -- W1 is the focus and W2 is the neighbor
        annotated_axes_neighbor = jnp.isin(jnp.arange(neighbor_rank), contraction_axes[:, 1])
        range_select_neighbor = jnp.asarray(neighbor_shape, dtype=int)[~annotated_axes_neighbor]
        output_filter_dim = 3

        def forward_neighbors_dot_general(indices: jax.Array) -> tuple[jax.Array, jax.Array]:
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
                    contraction_axes[0, 1],
                    indices[output_filter_dim],
                    neighbor_shape[contraction_axes[0, 1]],
                    focus_shape[output_filter_dim],
                ],
                dtype=int,
            )
            return self_slice, neighbor_slice

    return forward_neighbors_dot_general
