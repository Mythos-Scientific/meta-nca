import functools
import logging
from typing import Callable, Sequence

import chex
import jax

import metanca

from ._loss import compute_loss
from ._update_tasknet_n_steps import update_tasknet_n_steps
from .callbacks import CallbackRunner, TrainingContext
from .data_utils import promote_image_batch

logger = logging.getLogger(__name__)


@functools.partial(
    jax.jit,
    static_argnames=(
        "adj",
        "param_names",
        "n",
        "hidden_dim",
        "prop_cells_updated",
        "weight_transformer_dropout",
        "n_spatial_dims",
        "local_rule_net_apply",
    ),
)
def _update_tasknet_n_steps_jit(
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
    """JIT-compiled wrapper for update_tasknet_n_steps used in validation."""
    return update_tasknet_n_steps(
        local_rule_net_apply=local_rule_net_apply,
        local_rule_net_params=local_rule_net_params,
        rand_key=rand_key,
        tasknet_data=tasknet_data,
        hidden_dim=hidden_dim,
        param_names=param_names,
        adj=adj,
        n=n,
        prop_cells_updated=prop_cells_updated,
        weight_transformer_dropout=weight_transformer_dropout,
        n_spatial_dims=n_spatial_dims,
    )


def metanca_validation_step(
    val_batches: chex.Array,
    local_rule_net_apply: Callable[..., jax.Array],
    local_rule_params: chex.ArrayTree,
    rand_key: chex.PRNGKey,
    val_tasknet: metanca.TaskNet,
    n_update_steps: int,
    hidden_dim: int,
    prop_cells_updated: float,
    weight_transformer_dropout: float,
    callback_runner: CallbackRunner,
    n_spatial_dims: int = 2,
) -> tuple[dict[str, float], CallbackRunner]:
    """Run validation on all validation batches using a fresh tasknet.

    The tasknet is reset and updated once at the beginning, then loss and
    accuracy are computed over all validation batches without further updates.

    Note: The old codebase computed validation accuracy as the best across
    multiple random initializations and used early stopping to pick the best
    epoch (NCA update step) within each trajectory based on lowest train loss.
    This implementation evaluates after a fixed number of update steps with a
    single initialization, so reported validation accuracies may differ slightly
    from the old code, but the difference should not be large.

    Args:
        val_batches: Validation data batches (images, labels, masks).
        local_rule_params: Current local rule parameters.
        rand_key: Random key for tasknet reset and update steps.
        val_tasknet: TaskNet to use for validation (will be reset).
        n_update_steps: Number of local rule update steps to run.
        hidden_dim: Hidden dimension for the local rule.
        prop_cells_updated: Proportion of cells to update (same as training).
        weight_transformer_dropout: Dropout rate for weight transformer (same as training).

    Returns:
        tuple[dict[str, float], CallbackRunner]
    """
    import jax.numpy as jnp

    n_val_batches = val_batches[0].shape[0]

    # Reset the validation tasknet with a fresh random key
    rand_key, reset_key, update_key = jax.random.split(rand_key, 3)

    val_tasknet = val_tasknet.reset(reset_key)

    # Get tasknet data and static info
    tasknet_data = (
        val_tasknet.params,
        val_tasknet.hidden_states,
        val_tasknet.positional_encodings,
    )
    adj = val_tasknet.adj
    param_names = tuple(val_tasknet.iter_param_names())
    apply_fn = val_tasknet.model.apply

    init_param_leaves = jax.tree_util.tree_leaves(tasknet_data[0])
    init_hs_leaves = jax.tree_util.tree_leaves(tasknet_data[1])
    init_pe_leaves = jax.tree_util.tree_leaves(tasknet_data[2])

    logger.debug(
        f"Params[0] shape: {init_param_leaves[0].shape}, "
        f"mean={float(init_param_leaves[0].mean()):.4f}, "
        f"std={float(init_param_leaves[0].std()):.4f}"
    )
    logger.debug(
        f"HiddenState[0] shape: {init_hs_leaves[0].shape}, "
        f"mean={float(init_hs_leaves[0].mean()):.4f}, "
        f"std={float(init_hs_leaves[0].std()):.4f}"
    )
    logger.debug(
        f"PosEnc[0] shape: {init_pe_leaves[0].shape}, "
        f"mean={float(init_pe_leaves[0].mean()):.4f}, "
        f"std={float(init_pe_leaves[0].std()):.4f}"
    )
    logger.debug(f"HS == PE (layer 0): {bool(jnp.allclose(init_hs_leaves[0], init_pe_leaves[0]))}")

    # Update the tasknet once at the beginning
    updated_params, updated_hidden, _ = _update_tasknet_n_steps_jit(
        local_rule_net_apply=local_rule_net_apply,
        local_rule_net_params=local_rule_params,
        rand_key=update_key,
        tasknet_data=tasknet_data,
        hidden_dim=hidden_dim,
        param_names=param_names,
        adj=adj,
        n=n_update_steps,
        prop_cells_updated=prop_cells_updated,
        weight_transformer_dropout=weight_transformer_dropout,
        n_spatial_dims=n_spatial_dims,
    )

    # DEBUG: Check params after local rule updates
    logger.debug("\n--- After Local Rule Updates ---")
    updated_param_leaves = jax.tree_util.tree_leaves(updated_params)
    updated_hs_leaves = jax.tree_util.tree_leaves(updated_hidden)

    logger.debug(
        f"Params[0] mean={float(updated_param_leaves[0].mean()):.4f}, "
        f"std={float(updated_param_leaves[0].std()):.4f}"
    )
    logger.debug(
        f"HiddenState[0] mean={float(updated_hs_leaves[0].mean()):.4f}, "
        f"std={float(updated_hs_leaves[0].std()):.4f}"
    )

    # Check deltas
    param_delta = updated_param_leaves[0] - init_param_leaves[0]
    hs_delta = updated_hs_leaves[0] - init_hs_leaves[0]
    logger.debug(
        f"Param delta: mean={float(param_delta.mean()):.6f}, "
        f"std={float(param_delta.std()):.6f}, "
        f"max_abs={float(jnp.abs(param_delta).max()):.6f}"
    )
    logger.debug(
        f"HS delta: mean={float(hs_delta.mean()):.6f}, "
        f"std={float(hs_delta.std()):.6f}, "
        f"max_abs={float(jnp.abs(hs_delta).max()):.6f}"
    )

    # Check if local rule is actually changing params
    params_changed = not jnp.allclose(updated_param_leaves[0], init_param_leaves[0], atol=1e-6)
    logger.debug(f"Params changed after update: {params_changed}")
    logger.debug("=" * 60 + "\n")

    # Loop through validation batches and compute loss/accuracy

    ctx = TrainingContext(
        n_batches=n_val_batches,
        tasknet_data_list=[(updated_params, None, None)],
        apply_fns=[apply_fn],
    )

    # Loop through validation batches and compute metrics via callbacks
    metric_totals: dict[str, float] = {}

    for batch_idx in range(n_val_batches):
        batch_x_uint8, batch_y, mask = (
            val_batches[0][batch_idx],
            val_batches[1][batch_idx],
            val_batches[2][batch_idx],
        )
        batch_x = promote_image_batch(batch_x_uint8)

        loss = compute_loss(batch_x, batch_y, mask, updated_params, apply_fn)

        ctx = ctx.with_updated(
            batch_idx=batch_idx,
            batch_x=batch_x,
            batch_y=batch_y,
            mask=mask,
        )
        callback_runner, result = callback_runner.on_batch_end(ctx)

        metrics = result.logs | {"loss": loss}

        for metric_name, metric_value in metrics.items():
            metric_totals[metric_name] = metric_totals.get(metric_name, 0.0) + float(metric_value)

    return {
        metric: total / n_val_batches for metric, total in metric_totals.items()
    }, callback_runner
