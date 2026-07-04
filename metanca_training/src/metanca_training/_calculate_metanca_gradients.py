import functools
import logging
from typing import Callable, Sequence

import chex
import jax
import jax.numpy as jnp

import metanca

from ._clip_gradients import clip_gradients
from ._loss import compute_loss
from ._types import ApplyFn, Device
from ._update_tasknet_n_steps import update_tasknet_n_steps

logger = logging.getLogger(__name__)


@jax.jit
def _add_grads(a: chex.ArrayTree, b: chex.ArrayTree) -> chex.ArrayTree:
    """JIT'd grad accumulation — single XLA dispatch instead of per-leaf."""
    return jax.tree.map(jnp.add, a, b)


def _curry_div(divisor: chex.Scalar) -> Callable[[jax.Array], jax.Array]:
    @jax.jit
    def div(x: jax.Array) -> jax.Array:
        return x / divisor

    return div


@functools.partial(jax.value_and_grad, argnums=(0,), has_aux=True)
def _run_local_rule_forward_compute_tasknet_loss(
    local_rule_net_params: chex.ArrayTree,
    local_rule_net_apply: Callable[..., jax.Array],
    batch: tuple[jax.Array, jax.Array, jax.Array],
    n: int,
    tasknet_apply_fn: ApplyFn,
    rand_key: chex.PRNGKey,
    tasknet_data: tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree],
    hidden_dim: int,
    param_names: Sequence[str],
    adj: metanca.typing.AdjacencyDict,
    prop_cells_updated: float = 0.8,
    weight_transformer_dropout: float = 0.0,
    n_spatial_dims: int = 2,
) -> tuple[tuple[chex.ArrayTree, chex.ArrayTree], jax.Array]:
    xs, ys, mask = batch
    new_params, new_hidden_states, new_positional_encodings = update_tasknet_n_steps(
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

    batch_loss = compute_loss(xs, ys, mask, new_params, tasknet_apply_fn)

    return batch_loss, (new_params, new_hidden_states, new_positional_encodings)


# JIT'd version for multi-GPU: each arch gets its own JIT compilation,
# dispatched to a specific device via jax.device_put on inputs.
_jitted_run_local_rule = jax.jit(
    _run_local_rule_forward_compute_tasknet_loss,
    static_argnames=(
        "local_rule_net_apply",
        "n",
        "tasknet_apply_fn",
        "hidden_dim",
        "param_names",
        "adj",
        "prop_cells_updated",
        "weight_transformer_dropout",
        "n_spatial_dims",
    ),
)


def calculate_metanca_gradients(
    batches: Sequence[tuple[jax.Array, jax.Array, jax.Array]],
    local_rule_net_apply: Callable[..., jax.Array],
    local_rule_net_params: chex.ArrayTree,
    rand_key: chex.PRNGKey,
    tasknet_data_list: list[tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree]],
    tasknet_param_names: Sequence[Sequence[str]],
    tasknet_apply_fns: Sequence[ApplyFn],
    adjs: Sequence[metanca.typing.AdjacencyDict],
    n_update_steps: int,
    hidden_dim: int,
    prop_cells_updated: float = 0.8,
    weight_transformer_dropout: float = 0.0,
    grad_clip_norm: float = 1.0,
    n_spatial_dims: int = 2,
) -> tuple[chex.Scalar, chex.ArrayTree, list[tuple[chex.ArrayTree, chex.ArrayTree]]]:
    """Accumulate local-rule gradients across all tasknets.

    ``batches`` has one entry per tasknet (``batches[i]`` feeds tasknet ``i``),
    allowing mixed-vocab arch pools to draw each arch's batch from its own
    token stream. Classification callers pass the same batch repeated
    ``len(tasknet_data_list)`` times.
    """

    def _accumulate_metanca_gradients(
        batches: Sequence[tuple[jax.Array, jax.Array, jax.Array]],
        local_rule_net_apply: Callable[..., jax.Array],
        local_rule_net_params: chex.ArrayTree,
        rand_key: chex.PRNGKey,
    ) -> tuple[chex.ArrayTree, chex.Scalar, list[tuple[chex.ArrayTree, chex.ArrayTree]]]:

        accum_loss = 0.0
        accum_grads = jax.tree.map(jnp.zeros_like, local_rule_net_params)

        tasknet_rand_keys = jax.random.split(rand_key, len(tasknet_data_list))

        iterator = zip(
            tasknet_rand_keys,
            tasknet_data_list,
            adjs,
            tasknet_param_names,
            tasknet_apply_fns,
            batches,
        )
        new_tasknet_data_list = []
        for i, (rk, tasknet_data, adj, param_names, tasknet_apply_fn, batch) in enumerate(
            iterator, 1
        ):
            logger.debug(f"Accumulating model {i}/{len(tasknet_data_list)} grads")
            (model_loss, new_tasknet_data), (local_rule_grads,) = (
                _run_local_rule_forward_compute_tasknet_loss(
                    local_rule_net_params,
                    local_rule_net_apply,
                    batch=batch,
                    n=n_update_steps,
                    rand_key=rk,
                    tasknet_data=tasknet_data,
                    tasknet_apply_fn=tasknet_apply_fn,
                    hidden_dim=hidden_dim,
                    param_names=param_names,
                    adj=adj,
                    prop_cells_updated=prop_cells_updated,
                    weight_transformer_dropout=weight_transformer_dropout,
                    n_spatial_dims=n_spatial_dims,
                )
            )
            new_tasknet_data_list.append(new_tasknet_data)
            local_rule_grads = clip_gradients(local_rule_grads, grad_clip_norm=grad_clip_norm)
            accum_grads = jax.tree_util.tree_map(jnp.add, accum_grads, local_rule_grads)
            accum_loss += model_loss
            logger.debug(f"Model {i}/{len(tasknet_data_list)} updated.")

        accum_grads = jax.tree_util.tree_map(_curry_div(len(tasknet_data_list)), accum_grads)
        accum_loss = accum_loss / (len(tasknet_data_list))

        return accum_grads, accum_loss, new_tasknet_data_list

    accum_grads, accum_loss, new_tasknet_data_list = _accumulate_metanca_gradients(
        batches, local_rule_net_apply, local_rule_net_params, rand_key
    )

    return accum_loss, accum_grads, new_tasknet_data_list


def stage_data_to_devices(
    tasknet_data_list: list[tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree]],
    devices: Sequence[Device],
) -> list[tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree]]:
    """Pre-copy tasknet data to target devices. Call once per metaepoch."""
    return [jax.device_put(td, devices[i % len(devices)]) for i, td in enumerate(tasknet_data_list)]


def calculate_metanca_gradients_multi_gpu(
    batches: Sequence[tuple[jax.Array, jax.Array, jax.Array]],
    local_rule_net_apply: Callable[..., jax.Array],
    lr_params_per_device: list[chex.ArrayTree],
    rand_key: chex.PRNGKey,
    tasknet_data_list: list[tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree]],
    tasknet_param_names: Sequence[Sequence[str]],
    tasknet_apply_fns: Sequence[ApplyFn],
    adjs: Sequence[metanca.typing.AdjacencyDict],
    n_update_steps: int,
    hidden_dim: int,
    devices: Sequence[Device],
    prop_cells_updated: float = 0.8,
    weight_transformer_dropout: float = 0.0,
    n_spatial_dims: int = 2,
) -> tuple[chex.Scalar, list[chex.ArrayTree], list[tuple[chex.ArrayTree, chex.ArrayTree]]]:
    """Dispatch each arch to a different GPU and gather gradients.

    Args:
        lr_params_per_device: Pre-staged local rule params, one per device.
        tasknet_data_list: Pre-staged tasknet data, already on target devices.

    Returns raw list of per-arch grads (not accumulated) — the caller's JIT
    handles accumulation + averaging + clipping + optimizer in one XLA program.
    """
    import time as _time

    n_archs = len(tasknet_data_list)
    tasknet_rand_keys = jax.random.split(rand_key, n_archs)

    # Phase 1: Dispatch all archs to their target GPUs (async).
    # Data is pre-staged on devices — only batch and rand_key need transfer.
    t0 = _time.time()
    results = []
    iterator = zip(
        tasknet_rand_keys, tasknet_data_list, adjs, tasknet_param_names, tasknet_apply_fns
    )
    for i, (rk, tasknet_data, adj, param_names, tasknet_apply_fn) in enumerate(iterator):
        device = devices[i % len(devices)]
        batch_dev = jax.device_put(batches[i], device)
        rk_dev = jax.device_put(rk, device)

        result = _jitted_run_local_rule(
            lr_params_per_device[i % len(devices)],
            local_rule_net_apply=local_rule_net_apply,
            batch=batch_dev,
            n=n_update_steps,
            rand_key=rk_dev,
            tasknet_data=tasknet_data,
            tasknet_apply_fn=tasknet_apply_fn,
            hidden_dim=hidden_dim,
            param_names=param_names,
            adj=adj,
            prop_cells_updated=prop_cells_updated,
            weight_transformer_dropout=weight_transformer_dropout,
            n_spatial_dims=n_spatial_dims,
        )
        results.append(result)
    t_dispatch = _time.time() - t0

    # Phase 2: Gather grads + params on first device.
    # Only transfer grads (~2.2MB) and params (~3MB) — leave hidden_states
    # and PE on source device (not needed by callbacks, discarded on reset).
    t1 = _time.time()
    target_device = devices[0]
    all_grads = []
    accum_loss = 0.0
    new_tasknet_data_list = []

    for i, ((model_loss, new_tasknet_data), (local_rule_grads,)) in enumerate(results):
        local_rule_grads = jax.device_put(local_rule_grads, target_device)
        model_loss = jax.device_put(model_loss, target_device)
        all_grads.append(local_rule_grads)
        accum_loss += model_loss
        new_params, new_hs, new_pe = new_tasknet_data
        new_params = jax.device_put(new_params, target_device)
        new_tasknet_data_list.append((new_params, new_hs, new_pe))
    t_gather = _time.time() - t1

    accum_loss = accum_loss / n_archs

    logger.debug(f"multi-gpu: dispatch={t_dispatch*1000:.0f}ms " f"gather={t_gather*1000:.0f}ms")

    return accum_loss, all_grads, new_tasknet_data_list
