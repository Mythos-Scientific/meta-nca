"""Shared function for updating tasknet weights via local rule."""

from typing import Callable, Literal, Sequence

import chex
import jax

import metanca


def update_tasknet_n_steps(
    local_rule_net_apply: Callable[..., jax.Array],
    local_rule_net_params: chex.ArrayTree,
    rand_key: chex.PRNGKey,
    tasknet_data: tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree],
    hidden_dim: int,
    param_names: Sequence[str],
    adj: metanca.typing.AdjacencyDict,
    n: int,
    prop_cells_updated: float = 0.8,
    weight_transformer_dropout: float = 0.0,
    n_spatial_dims: int = 2,
) -> tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree]:
    """Update tasknet parameters n times using the local rule.

    This function is used by both training and validation to ensure
    consistent behavior.

    Args:
        local_rule_net_params: Parameters of the local rule network.
        rand_key: Random key for stochastic updates.
        tasknet_data: Tuple of (params, hidden_states, positional_encodings).
        hidden_dim: Hidden dimension for the local rule.
        param_names: Names of parameters to update.
        adj: Adjacency dictionary describing parameter neighborhoods.
        n: Number of update steps to run.
        prop_cells_updated: Proportion of cells to update each step.
        weight_transformer_dropout: Dropout rate for weight transformer.
        n_spatial_dims: Number of spatial dimensions.

    Returns:
        Tuple of (updated_params, updated_hidden_states, positional_encodings).
    """

    @jax.checkpoint
    def _scan_body(
        carry: tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree], rk: chex.PRNGKey
    ) -> tuple[tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree], Literal[None]]:
        params, hidden_states, positional_encodings = carry

        next_params, next_hidden = metanca.update_tasknet(
            local_rule_net_apply=local_rule_net_apply,
            local_rule_net_params=local_rule_net_params,
            params=params,
            hidden_states=hidden_states,
            positional_encodings=positional_encodings,
            rand_key=rk,
            adj=adj,
            hidden_dim=hidden_dim,
            param_names=param_names,
            prop_cells_updated=prop_cells_updated,
            weight_transformer_dropout=weight_transformer_dropout,
            n_spatial_dims=n_spatial_dims,
        )
        return (next_params, next_hidden, positional_encodings), None

    update_keys = jax.random.split(rand_key, n)

    (final_params, final_hidden, final_positional_encodings), _ = jax.lax.scan(
        _scan_body, tasknet_data, update_keys
    )
    return final_params, final_hidden, final_positional_encodings
