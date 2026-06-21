import functools
from typing import Callable, Sequence

import chex
import jax
import optax

import metanca

from ._calculate_metanca_gradients import (
    calculate_metanca_gradients,
    calculate_metanca_gradients_multi_gpu,
)
from ._metanca_optimizer_step import metanca_optimizer_step
from ._types import ApplyFn, Device


@functools.partial(
    jax.jit,
    static_argnames=(
        "adjs",
        "tasknet_param_names",
        "n_update_steps",
        "hidden_dim",
        "weight_transformer_dropout",
        "prop_cells_updated",
        "optimizer",
        "tasknet_apply_fns",
        "grad_clip_norm",
        "n_spatial_dims",
        "local_rule_net_apply",
    ),
)
def metanca_train_step(
    batch: tuple[jax.Array, jax.Array, jax.Array],
    local_rule_net_apply: Callable[..., jax.Array],
    local_rule_params: chex.ArrayTree,
    rand_key: chex.PRNGKey,
    tasknet_data_list: list[tuple[chex.ArrayTree, chex.ArrayTree]],
    adjs: Sequence[metanca.typing.AdjacencyDict],
    tasknet_param_names: Sequence[Sequence[str]],
    tasknet_apply_fns: Sequence[ApplyFn],
    n_update_steps: int,
    hidden_dim: int,
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
    weight_transformer_dropout: float = 0.0,
    prop_cells_updated: float = 0.8,
    grad_clip_norm: float = 1.0,
    n_spatial_dims: int = 2,
) -> tuple[
    list[tuple[chex.ArrayTree, chex.ArrayTree]],
    chex.Scalar,
    chex.ArrayTree,
    optax.OptState,
    chex.PRNGKey,
]:

    rand_key, train_key = jax.random.split(rand_key)

    avg_loss, local_rule_grads, new_tasknet_data_list = calculate_metanca_gradients(
        batch=batch,
        local_rule_net_apply=local_rule_net_apply,
        local_rule_net_params=local_rule_params,
        rand_key=train_key,
        tasknet_data_list=tasknet_data_list,
        tasknet_apply_fns=tasknet_apply_fns,
        adjs=adjs,
        n_update_steps=n_update_steps,
        hidden_dim=hidden_dim,
        tasknet_param_names=tasknet_param_names,
        prop_cells_updated=prop_cells_updated,
        weight_transformer_dropout=weight_transformer_dropout,
        grad_clip_norm=grad_clip_norm,
        n_spatial_dims=n_spatial_dims,
    )

    local_rule_params, optimizer_state = metanca_optimizer_step(
        local_rule_params, local_rule_grads, optimizer, optimizer_state
    )

    return new_tasknet_data_list, avg_loss, local_rule_params, optimizer_state, rand_key


@functools.partial(jax.jit, static_argnames=("optimizer", "n_archs", "grad_clip_norm"))
def _jitted_accumulate_and_optimize(
    all_grads: list[chex.ArrayTree],
    n_archs: int,
    grad_clip_norm: float,
    local_rule_params: chex.ArrayTree,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[chex.ArrayTree, optax.OptState]:
    """JIT'd: accumulate + average + clip + optimizer in one XLA program.

    all_grads is a list of per-arch gradient pytrees (same structure).
    The list length becomes part of the JIT trace cache key.
    """
    from ._clip_gradients import clip_gradients

    clipped = [clip_gradients(g, grad_clip_norm=grad_clip_norm) for g in all_grads]
    accum = jax.tree.map(lambda *gs: sum(gs), *clipped)
    grads = jax.tree.map(lambda g: g / n_archs, accum)
    return metanca_optimizer_step(local_rule_params, grads, optimizer, opt_state)


def metanca_train_step_multi_gpu(
    batch: tuple[jax.Array, jax.Array, jax.Array],
    local_rule_net_apply: Callable[..., jax.Array],
    local_rule_params: chex.ArrayTree,
    lr_params_per_device: list[chex.ArrayTree],
    rand_key: chex.PRNGKey,
    tasknet_data_list: list[tuple[chex.ArrayTree, chex.ArrayTree]],
    adjs: Sequence[metanca.typing.AdjacencyDict],
    tasknet_param_names: Sequence[Sequence[str]],
    tasknet_apply_fns: Sequence[ApplyFn],
    n_update_steps: int,
    hidden_dim: int,
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
    devices: Sequence[Device],
    weight_transformer_dropout: float = 0.0,
    prop_cells_updated: float = 0.8,
    grad_clip_norm: float = 1.0,
    n_spatial_dims: int = 2,
) -> tuple[
    list[tuple[chex.ArrayTree, chex.ArrayTree]],
    chex.Scalar,
    chex.ArrayTree,
    list[chex.ArrayTree],
    optax.OptState,
    chex.PRNGKey,
]:
    """Multi-GPU train step — pure Python orchestrator (not JIT'd).

    Args:
        lr_params_per_device: Pre-staged local rule params per device.
        tasknet_data_list: Pre-staged tasknet data on target devices.

    Returns:
        (new_tasknet_data_list, avg_loss, local_rule_params,
         lr_params_per_device, optimizer_state, rand_key)
    """
    rand_key, train_key = jax.random.split(rand_key)

    avg_loss, all_grads, new_tasknet_data_list = calculate_metanca_gradients_multi_gpu(
        batch=batch,
        local_rule_net_apply=local_rule_net_apply,
        lr_params_per_device=lr_params_per_device,
        rand_key=train_key,
        tasknet_data_list=tasknet_data_list,
        tasknet_apply_fns=tasknet_apply_fns,
        adjs=adjs,
        n_update_steps=n_update_steps,
        hidden_dim=hidden_dim,
        tasknet_param_names=tasknet_param_names,
        devices=devices,
        prop_cells_updated=prop_cells_updated,
        weight_transformer_dropout=weight_transformer_dropout,
        n_spatial_dims=n_spatial_dims,
    )

    # Accumulate + average + clip + optimize in one JIT'd call
    local_rule_params, optimizer_state = _jitted_accumulate_and_optimize(
        all_grads,
        n_archs=len(tasknet_data_list),
        grad_clip_norm=grad_clip_norm,
        local_rule_params=local_rule_params,
        optimizer=optimizer,
        opt_state=optimizer_state,
    )

    # Update pre-staged lr_params on all devices
    lr_params_per_device = [jax.device_put(local_rule_params, d) for d in devices]

    return (
        new_tasknet_data_list,
        avg_loss,
        local_rule_params,
        lr_params_per_device,
        optimizer_state,
        rand_key,
    )
