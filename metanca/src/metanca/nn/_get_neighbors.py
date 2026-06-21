from typing import Literal, Sequence

import chex
import jax

import metanca.functional as mfx
import metanca.strategies as msx
from metanca.typing import AdjacencyDict, NeighborNeuronViewDict, NeuronView, SliceInfo


def _build_neuron_view(
    param: jax.Array,
    hidden_state: jax.Array,
    positional_encoding: jax.Array,
    slice_info: SliceInfo,
    rotation_source: jax.Array = None,
) -> NeuronView:
    """Build a NeuronView from parameter, hidden state, and positional encoding arrays.

    Args:
        param: Weight parameter array.
        hidden_state: Hidden state array.
        positional_encoding: Positional encoding array.
        slice_info: Slice information for reshaping.
        rotation_source: Optional array to use for positional_encodings in the view.
            If None, uses positional_encoding. This is used to match the original code's
            behavior where backward neighbors use hidden_states for rotary embeddings
            instead of positional_encodings.

    Returns:
        NeuronView dictionary.
    """

    if slice_info["slice_size"] != 1:
        stride = slice_info["stride"]
    else:
        stride = None

    rotation_array = rotation_source if rotation_source is not None else positional_encoding

    return {
        "weights": mfx.swap_axes_and_reshape(param, slice_info["index_dim"], 1, stride=stride),
        "hidden_states": mfx.swap_axes_and_reshape(
            hidden_state, slice_info["index_dim"], hidden_state.shape[-1], stride=stride
        ),
        "positional_encodings": mfx.swap_axes_and_reshape(
            rotation_array,
            slice_info["index_dim"],
            rotation_array.shape[-1],
            stride=stride,
        ),
    }


def _create_reflexive_neuron_view(
    params: chex.ArrayTree,
    hidden_states: chex.ArrayTree,
    positional_encodings: chex.ArrayTree,
    hidden_dim: int,
    param_shape: Sequence[int],
    param_name: str,
    param_type: Literal["bias", "kernel"],
    direction: Literal["fwd", "bwd"],
) -> tuple[NeuronView, int]:
    focus_nv = {}

    match (param_type, direction):
        case ("kernel", "fwd"):
            focus_index_dim = len(param_shape) - 1
        case ("kernel", "bwd"):
            focus_index_dim = len(param_shape) - 2

        case ("bias", _):
            focus_index_dim = 0

    nv_attrs = ("weights", "hidden_states", "positional_encodings")
    tasknet_attrs = (params, hidden_states, positional_encodings)

    for i, (nv_attr, tasknet_attr) in enumerate(zip(nv_attrs, tasknet_attrs)):

        if i == 0:
            dimension = 1
        else:
            dimension = hidden_dim

        item = mfx.nested_get(param_name, tasknet_attr)
        item_nv = mfx.swap_axes_and_reshape(item, focus_index_dim, dimension)
        focus_nv[nv_attr] = item_nv

    return focus_nv, focus_index_dim


def _create_reflexive_neighbor_view(
    focus_nv: NeuronView,
    slice_info: SliceInfo,
    direction: Literal["fwd", "bwd"],
    param_name: str,
) -> NeighborNeuronViewDict:
    return {
        "direction": direction,
        "neighbor": focus_nv,
        "focus": focus_nv,
        "slice_info": {
            # "focus": {"index_dim": index_dim, "slice_size": 1}
            "focus": slice_info
        },
        "neighbor_name": param_name,
    }


def _get_neuron_oriented_neighbor_views(
    focus_parameter_name: str,
    adj: dict[Literal["fwd", "bwd"], AdjacencyDict],
    params: chex.ArrayTree,
    positional_encodings: chex.ArrayTree,
    hidden_states: chex.ArrayTree,
    direction: Literal["fwd", "bwd"],
) -> tuple[NeuronView, SliceInfo, list[NeighborNeuronViewDict]]:

    focus_parameter = mfx.nested_get(focus_parameter_name, params)
    focus_hidden_state = mfx.nested_get(focus_parameter_name, hidden_states)
    focus_positional_encoding = mfx.nested_get(focus_parameter_name, positional_encodings)

    realm = {"fwd": "forward_slice_info", "bwd": "backward_slice_info"}[direction]

    # at the parameter level, one, but not both,
    # parameter neighborhood may be empty.
    # excluding biases, most parameters somehow
    # self interact since weights share input/output neurons.

    focus_nv, focus_index_dim = _create_reflexive_neuron_view(
        params,
        hidden_states,
        positional_encodings,
        hidden_dim=focus_hidden_state.shape[-1],
        param_shape=focus_parameter.shape,
        param_name=focus_parameter_name,
        param_type=focus_parameter_name.split(".")[-1],
        direction=direction,
    )
    focus_slice_info = {"slice_size": 1, "index_dim": focus_index_dim, "stride": None}
    reflexive_focus_nv = focus_nv
    reflexive_slice_info = focus_slice_info

    neuron_oriented_views = []
    conv_dense_slice_info = None
    conv_dense_neuron_count = None
    if focus_parameter_name in adj[direction]:
        for neighbor_parameter_name, msg_fn, msg_fn_params in adj[direction][focus_parameter_name]:

            # print(msg_fn, direction, neighbor_parameter_name, focus_parameter_name)

            neighbor_parameter = mfx.nested_get(neighbor_parameter_name, params)
            neighbor_hidden_state = mfx.nested_get(neighbor_parameter_name, hidden_states)
            neighbor_positional_encoding = mfx.nested_get(
                neighbor_parameter_name, positional_encodings
            )

            slice_info = msx.apply_strategy(realm, msg_fn, **msg_fn_params)
            if slice_info is None:
                raise RuntimeError(
                    "No slice strategy output for neighbor pair: "
                    f"direction={direction}, realm={realm}, msg_fn={msg_fn}, "
                    f"focus={focus_parameter_name}, neighbor={neighbor_parameter_name}, "
                    f"msg_fn_params={msg_fn_params}"
                )

            # For backward direction, use positional_encodings (not hidden_states).
            rotation_source = None

            neighbor_nv = _build_neuron_view(
                neighbor_parameter,
                neighbor_hidden_state,
                neighbor_positional_encoding,
                slice_info["neighbor"],
                rotation_source=rotation_source,
            )

            if (
                direction == "bwd"
                and focus_parameter.ndim == 2
                and (
                    neighbor_parameter.ndim > 2
                    or ("Conv" in neighbor_parameter_name and neighbor_parameter.ndim == 1)
                )
            ):
                conv_dense_slice_info = slice_info
                conv_dense_neuron_count = neighbor_nv["weights"].shape[0]
            neuron_oriented_views.append(
                {
                    "direction": direction,
                    "neighbor_name": neighbor_parameter_name,
                    "neighbor": neighbor_nv,
                    "slice_info": slice_info,
                }
            )

    if conv_dense_slice_info is not None and conv_dense_neuron_count is not None:
        reflexive_neuron_count = reflexive_focus_nv["weights"].shape[0]
        if conv_dense_neuron_count != reflexive_neuron_count:
            reflexive_focus_nv = _build_neuron_view(
                focus_parameter,
                focus_hidden_state,
                focus_positional_encoding,
                conv_dense_slice_info["focus"],
            )
            reflexive_slice_info = conv_dense_slice_info["focus"]
            focus_nv = reflexive_focus_nv
            focus_slice_info = reflexive_slice_info

    neuron_oriented_views = [
        _create_reflexive_neighbor_view(
            reflexive_focus_nv, reflexive_slice_info, direction, focus_parameter_name
        )
    ] + neuron_oriented_views
    return reflexive_focus_nv, reflexive_slice_info, neuron_oriented_views


def get_neighbors(
    focus_parameter_name: str,
    adj: AdjacencyDict,
    params: dict,
    positional_encodings: dict,
    hidden_states: dict,
) -> dict[Literal["fwd", "bwd"], tuple[NeuronView, list[NeighborNeuronViewDict]]]:
    def get_views(
        direction: Literal["fwd", "bwd"],
    ) -> tuple[NeuronView, SliceInfo, list[NeighborNeuronViewDict]]:
        return _get_neuron_oriented_neighbor_views(
            focus_parameter_name=focus_parameter_name,
            adj=adj,
            params=params,
            positional_encodings=positional_encodings,
            hidden_states=hidden_states,
            direction=direction,
        )

    return {d: get_views(d) for d in ("fwd", "bwd")}
