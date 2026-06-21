import functools
import logging
from typing import Callable, Literal, Sequence

import chex
import jax
import jax.numpy as jnp

import metanca.functional as mfx
from metanca.typing import AdjacencyDict, NeighborNeuronViewDict, NeuronView, SliceInfo

from ._get_neighbors import get_neighbors

logger = logging.getLogger(__name__)


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


# @functools.partial(
#    jax.jit,
#    static_argnames=(
#        "adj",
#        "hidden_dim",
#        "param_names",
#        "prop_cells_updated",
#        "weight_transformer_dropout",
#    ),
# )
def update_tasknet(
    params: chex.ArrayTree,
    hidden_states: chex.ArrayTree,
    positional_encodings: chex.ArrayTree,
    local_rule_net_apply: Callable[..., jax.Array],
    local_rule_net_params: chex.ArrayTree,
    rand_key: chex.PRNGKey,
    adj: dict[Literal["fwd", "bwd"], AdjacencyDict],
    hidden_dim: int,
    param_names: Sequence[str],
    prop_cells_updated: float = 0.8,
    weight_transformer_dropout: float = 0.8,
    n_spatial_dims: int = 2,
) -> tuple[chex.ArrayTree, chex.ArrayTree]:
    del n_spatial_dims
    new_tasknet_params, new_tasknet_hidden_states = {}, {}

    @functools.partial(jax.checkpoint, static_argnums=(0,))
    def _for_body(param_name: str, rand_key: chex.PRNGKey):
        param = mfx.nested_get(param_name, params)
        param_shape = param.shape
        *_, param_type = param_name.split(".")

        rand_key, param_mask_key = jax.random.split(rand_key)
        param_mask = mfx.get_weight_mask(param_mask_key, param, prop_cells_updated)
        num_cells_updated = int(prop_cells_updated * param.size)

        neighbors = get_neighbors(
            param_name,
            params=params,
            positional_encodings=positional_encodings,
            hidden_states=hidden_states,
            adj=adj,
        )

        focus_nv_fwd, fwd_slice_info, fwd_neighbor_nvs = neighbors["fwd"]
        focus_nv_bwd, bwd_slice_info, bwd_neighbor_nvs = neighbors["bwd"]

        fwd_stride = fwd_slice_info["stride"]
        bwd_stride = bwd_slice_info["stride"]

        param_mask_fwd_nv = mfx.swap_axes_and_reshape(
            param_mask, fwd_slice_info["index_dim"], 1, stride=fwd_stride
        ).squeeze(-1)
        param_mask_bwd_nv = mfx.swap_axes_and_reshape(
            param_mask, bwd_slice_info["index_dim"], 1, stride=bwd_stride
        ).squeeze(-1)

        focus_vectors_fwd = jnp.concatenate(
            [focus_nv_fwd["weights"], focus_nv_fwd["hidden_states"]], axis=-1
        )
        focus_vectors_bwd = jnp.concatenate(
            [focus_nv_bwd["weights"], focus_nv_bwd["hidden_states"]], axis=-1
        )
        focus_pos_enc_fwd = focus_nv_fwd["positional_encodings"]
        focus_pos_enc_bwd = focus_nv_bwd["positional_encodings"]

        if fwd_neighbor_nvs:
            neighbor_vectors_fwd = jnp.concatenate(
                [
                    jnp.concatenate(
                        [view_dict["neighbor"]["weights"], view_dict["neighbor"]["hidden_states"]],
                        axis=-1,
                    )
                    for view_dict in fwd_neighbor_nvs
                ],
                axis=1,
            )
            neighbor_pos_enc_fwd = jnp.concatenate(
                [view_dict["neighbor"]["positional_encodings"] for view_dict in fwd_neighbor_nvs],
                axis=1,
            )
        else:
            neighbor_vectors_fwd = jnp.zeros(
                (focus_vectors_fwd.shape[0], 0, focus_vectors_fwd.shape[-1]),
                dtype=focus_vectors_fwd.dtype,
            )
            neighbor_pos_enc_fwd = jnp.zeros(
                (focus_vectors_fwd.shape[0], 0, focus_pos_enc_fwd.shape[-1]),
                dtype=focus_pos_enc_fwd.dtype,
            )

        if bwd_neighbor_nvs:
            neighbor_vectors_bwd = jnp.concatenate(
                [
                    jnp.concatenate(
                        [view_dict["neighbor"]["weights"], view_dict["neighbor"]["hidden_states"]],
                        axis=-1,
                    )
                    for view_dict in bwd_neighbor_nvs
                ],
                axis=1,
            )
            neighbor_pos_enc_bwd = jnp.concatenate(
                [view_dict["neighbor"]["positional_encodings"] for view_dict in bwd_neighbor_nvs],
                axis=1,
            )
        else:
            neighbor_vectors_bwd = jnp.zeros(
                (focus_vectors_bwd.shape[0], 0, focus_vectors_bwd.shape[-1]),
                dtype=focus_vectors_bwd.dtype,
            )
            neighbor_pos_enc_bwd = jnp.zeros(
                (focus_vectors_bwd.shape[0], 0, focus_pos_enc_bwd.shape[-1]),
                dtype=focus_pos_enc_bwd.dtype,
            )

        deterministic = weight_transformer_dropout == 0.0
        rngs = None if deterministic else {"dropout": rand_key}
        delta_theta, delta_hidden, fwd_adjusted_inds_i, fwd_adjusted_inds_j = local_rule_net_apply(
            local_rule_net_params,
            focus_vectors_fwd,
            focus_vectors_bwd,
            neighbor_vectors_fwd,
            neighbor_vectors_bwd,
            focus_pos_enc_fwd,
            focus_pos_enc_bwd,
            neighbor_pos_enc_fwd,
            neighbor_pos_enc_bwd,
            param_mask_fwd_nv,
            param_mask_bwd_nv,
            num_cells_updated,
            param_shape=param_shape,
            fwd_index_dim=fwd_slice_info["index_dim"],
            fwd_stride=fwd_stride,
            bwd_index_dim=bwd_slice_info["index_dim"],
            bwd_stride=bwd_stride,
            deterministic=deterministic,
            rngs=rngs,
        )

        updated_weight_nv = (
            focus_nv_fwd["weights"]
            .at[fwd_adjusted_inds_i, fwd_adjusted_inds_j]
            .add(delta_theta[:, None])
        )
        updated_state_nv = (
            focus_nv_fwd["hidden_states"]
            .at[fwd_adjusted_inds_i, fwd_adjusted_inds_j]
            .add(delta_hidden)
        )

        updated_weight = mfx.invert_swap_axes_and_reshape(
            updated_weight_nv, param_shape, fwd_slice_info["index_dim"], stride=fwd_stride
        )
        updated_state = mfx.invert_swap_axes_and_reshape(
            updated_state_nv,
            (*param_shape, hidden_dim),
            fwd_slice_info["index_dim"],
            stride=fwd_stride,
        )

        return updated_weight, updated_state, rand_key

    for param_name in param_names:
        logger.debug(f"running {param_name} update")
        updated_weight, updated_state, rand_key = _for_body(param_name, rand_key)
        mfx.nested_set(param_name, updated_weight, new_tasknet_params)
        mfx.nested_set(param_name, updated_state, new_tasknet_hidden_states)
        logger.debug(f"updated {param_name}!")

    return (new_tasknet_params, new_tasknet_hidden_states)
