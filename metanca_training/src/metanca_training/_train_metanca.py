import logging
import os
import random
import time

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import wandb
from etils import epath
from omegaconf import DictConfig

import metanca

from ._best_metrics import build_metrics_with_best, update_best_metrics
from ._calculate_metanca_gradients import stage_data_to_devices
from ._checkpointing import (
    build_checkpoint_metadata,
    create_checkpoint_manager,
    restore_checkpoint,
    save_checkpoint,
)
from ._config_factory import compute_hidden_state_dim, compute_local_rule_arch
from ._metanca_train_step import metanca_train_step, metanca_train_step_multi_gpu
from ._sample_pool import SamplePool
from ._training_vars import TrainingVars
from ._update_step_schedule import constant_update_step, increment_update_step
from ._validation_step import metanca_validation_step
from .callbacks import (
    CallbackRunner,
    TrainingContext,
    create_accuracy_callback,
    create_early_stopping,
)
from .data_utils import promote_image_batch
from .init_functions import init_local_rule_net

logger = logging.getLogger(__name__)


def _n_spatial_dims(input_shape: tuple[int, ...]) -> int:
    return len(input_shape) - 1


def before_metanca_training(
    cfg: DictConfig,
    *,
    models: list[nn.Module],
    test_model: nn.Module,
    rand_key: chex.PRNGKey,
) -> TrainingVars:
    input_shape = tuple(cfg.dataset.input_shape)
    n_spatial_dims = _n_spatial_dims(input_shape)
    local_rule_arch = compute_local_rule_arch(
        compute_hidden_state_dim(
            input_shape,
            cfg.positional_encoding.d_layer,
            cfg.positional_encoding.d_neuron,
            cfg.positional_encoding.d_spatial,
        ),
        list(cfg.local_rule.hidden_layers),
    )
    checkpoint_dir = (epath.Path(cfg.checkpoint.checkpoint_dir) / cfg.checkpoint.run_name).resolve()

    rand_key, tasknet_rand_key = jax.random.split(rand_key)
    *tasknets, test_tasknet = metanca.TaskNet.build_many(
        models=models + [test_model],
        input_shapes=[input_shape] * len(models) + [input_shape],
        key=tasknet_rand_key,
        d_neuron=cfg.positional_encoding.d_neuron,
        d_layer=cfg.positional_encoding.d_layer,
        d_spatial=cfg.positional_encoding.d_spatial,
    )

    # TODO: could configure this if we wanted.
    monitor = "accuracy"

    local_rule_net, local_rule_params = init_local_rule_net(
        local_rule_arch,
        test_tasknet.hidden_dim,
        rand_key,
        n_spatial_dims,
        cfg.training.weight_transformer_dropout,
    )

    optimizer = optax.contrib.muon(learning_rate=cfg.training.lr)

    checkpoint_metadata = build_checkpoint_metadata(
        "contrib.muon",
        {"learning_rate": cfg.training.lr},
    )
    opt_state = optimizer.init(local_rule_params)
    start_metaepoch = 0

    checkpoint_manager = create_checkpoint_manager(
        checkpoint_dir,
        cfg.checkpoint.save_top_n,
        monitor=monitor,
    )
    if (step := checkpoint_manager.latest_step()) is not None:
        logger.info(f"Restoring previous local rule state (step {step})")
        checkpoint_metadata, local_rule_params, optimizer, opt_state = restore_checkpoint(
            step, {"params": local_rule_params, "opt_state": opt_state}, checkpoint_manager
        )
        start_metaepoch = step

    n_update_steps = cfg.training.num_epochs
    match cfg.training.update_step_scheduler_type:
        case "constant":
            update_step_schedule = constant_update_step(n_update_steps)
        case "increment":
            n_update_steps = (start_metaepoch // cfg.training.update_step_scheduler_rate) + 1
            if cfg.training.update_step_scheduler_max_steps is not None:
                n_update_steps = min(n_update_steps, cfg.training.update_step_scheduler_max_steps)
            if n_update_steps > 1:
                logger.info(f"Fast-forwarded # update steps to {n_update_steps}")

            update_step_schedule = increment_update_step(
                cfg.training.update_step_scheduler_rate,
                cfg.training.update_step_scheduler_max_steps,
            )

    # Sample pooling (one pool per training architecture)
    pool_cfg = cfg.training.sample_pooling
    if pool_cfg.enabled:
        pools = [SamplePool(capacity=pool_cfg.pool_size) for _ in tasknets]
        logger.info(
            f"Sample pooling enabled: pool_size={pool_cfg.pool_size}, "
            f"seed_fraction={pool_cfg.seed_fraction}"
        )
    else:
        pools = []

    tvars = TrainingVars(
        tasknets=tasknets,
        test_tasknet=test_tasknet,
        local_rule_params=local_rule_params,
        local_rule_net_apply=local_rule_net.apply,
        rand_key=rand_key,
        optimizer=optimizer,
        optimizer_state=opt_state,
        hidden_dim=test_tasknet.hidden_dim,
        checkpoint_manager=checkpoint_manager,
        update_step_scheduler=update_step_schedule,
        n_update_steps=n_update_steps,
        checkpoint_metadata=checkpoint_metadata,
        start_metaepoch=start_metaepoch,
        pools=pools,
    )
    return tvars


def log_dict(
    metrics: dict[str, float],
    step: int,
    prefix: str = "",
    *,
    best_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    updated_best_metrics = update_best_metrics(best_metrics or {}, metrics)
    logged_metrics = build_metrics_with_best(metrics, updated_best_metrics)
    wandb.log(logged_metrics, step=step)
    metric_string = ", ".join(f"{name}={value:0.3f}" for name, value in logged_metrics.items())
    logger.info(f"{prefix} ({step=}): {metric_string}")
    return updated_best_metrics


def train_metanca(
    train_batches: chex.Array,
    val_batches: chex.Array,
    *,
    cfg: DictConfig,
    models: list[nn.Module],
    test_model: nn.Module,
):
    logger.info("Hydra training config:\n%s", cfg)
    input_shape = tuple(cfg.dataset.input_shape)
    n_spatial_dims = _n_spatial_dims(input_shape)
    training_vars = before_metanca_training(
        cfg,
        models=models,
        test_model=test_model,
        rand_key=jax.random.key(cfg.random.seed),
    )

    devices = jax.devices()
    use_multi_gpu = len(devices) > 1
    if use_multi_gpu:
        logger.info(f"Multi-GPU enabled: {len(devices)} devices ({devices})")
    else:
        logger.info(f"Single-GPU mode: {devices[0]}")

    tasknets = training_vars.tasknets
    pools = training_vars.pools
    adjs = tuple(tasknet.adj for tasknet in tasknets)
    tasknet_param_names = tuple(tuple(tasknet.iter_param_names()) for tasknet in tasknets)
    tasknet_apply_fns = tuple(tasknet.model.apply for tasknet in tasknets)

    hidden_dim = training_vars.hidden_dim

    n_train_batches = train_batches[0].shape[0]
    local_rule_params = training_vars.local_rule_params

    optimizer = training_vars.optimizer
    opt_state = training_vars.optimizer_state
    rand_key = jax.random.key(cfg.random.seed)

    callback_runner = CallbackRunner.create(
        [
            create_accuracy_callback(),
            create_early_stopping(
                monitor="val/loss",
                patience=getattr(cfg.training, "early_stopping_patience", 10),
                mode="min",
            ),
        ]
    )
    ctx = TrainingContext(
        n_batches=n_train_batches,
        local_rule_params=local_rule_params,
        optimizer_state=opt_state,
        apply_fns=tasknet_apply_fns,
    )
    callback_runner, _ = callback_runner.on_train_start(ctx)

    elapsed_times = jnp.zeros(cfg.logging.log_every_n_metaepochs)

    n_update_steps = training_vars.n_update_steps

    # Loss spike detection via EMA
    ema_loss = None
    loss_spike_rollbacks = 0
    consecutive_rollbacks = 0
    pool_flushes = 0
    spike_multiplier = cfg.training.sample_pooling.loss_spike_multiplier
    max_consecutive_rollbacks = cfg.training.sample_pooling.max_consecutive_rollbacks
    best_metrics: dict[str, float] = {}

    for metaepoch in range(training_vars.start_metaepoch, cfg.training.num_metaepochs):

        n_update_steps = training_vars.update_step_scheduler(metaepoch, n_update_steps)

        t_start = time.time()
        rand_key, tasknet_rand_key = jax.random.split(rand_key)
        reset_keys = jax.random.split(tasknet_rand_key, num=len(tasknets))

        if pools:
            seed_fraction = cfg.training.sample_pooling.seed_fraction
            for i, (tasknet, rk) in enumerate(zip(tasknets, reset_keys)):
                if pools[i].size > 0 and random.random() > seed_fraction:
                    params, hs = pools[i].sample()
                    tasknets[i] = tasknet.with_updated(
                        params=SamplePool.to_jax(params),
                        hidden_states=SamplePool.to_jax(hs),
                    )
                else:
                    tasknets[i] = tasknet.reset(rk)
        else:
            tasknets = [tasknet.reset(rk) for tasknet, rk in zip(tasknets, reset_keys)]

        tasknet_data_list = [
            (tasknet.params, tasknet.hidden_states, tasknet.positional_encodings)
            for tasknet in tasknets
        ]

        # Pre-stage data to devices once per metaepoch (avoids 340MB copy per batch)
        if use_multi_gpu:
            tasknet_data_list = stage_data_to_devices(tasknet_data_list, devices)
            lr_params_per_device = [jax.device_put(local_rule_params, d) for d in devices]

        metric_totals = {}
        batch_metrics = {}

        # Save pre-metaepoch state for NaN rollback
        metaepoch_local_rule_params = local_rule_params
        metaepoch_opt_state = opt_state
        metaepoch_rand_key = rand_key
        if use_multi_gpu:
            metaepoch_lr_params_per_device = lr_params_per_device

        # Shuffle batch indices like the original code
        rand_key, shuffle_key = jax.random.split(rand_key)
        shuffled_inds = jax.random.permutation(shuffle_key, jnp.arange(n_train_batches))

        timing_enabled = os.environ.get("METANCA_TIMING", "0") == "1"
        nan_detected = False

        for i, batch_idx in enumerate(shuffled_inds):
            t_start_batch = time.time()
            batch_x_uint8, batch_y, mask = (
                train_batches[0][batch_idx],
                train_batches[1][batch_idx],
                train_batches[2][batch_idx],
            )
            batch_x = promote_image_batch(batch_x_uint8)
            batch = (batch_x, batch_y, mask)
            t_after_prep = time.time()

            t_train_start = time.time()
            if use_multi_gpu:
                (
                    new_tasknet_data,
                    batch_avg_loss,
                    local_rule_params,
                    lr_params_per_device,
                    opt_state,
                    rand_key,
                ) = metanca_train_step_multi_gpu(
                    batch=batch,
                    local_rule_net_apply=training_vars.local_rule_net_apply,
                    local_rule_params=local_rule_params,
                    lr_params_per_device=lr_params_per_device,
                    rand_key=rand_key,
                    tasknet_data_list=tasknet_data_list,
                    adjs=adjs,
                    tasknet_param_names=tasknet_param_names,
                    tasknet_apply_fns=tasknet_apply_fns,
                    n_update_steps=n_update_steps,
                    hidden_dim=hidden_dim,
                    optimizer=optimizer,
                    optimizer_state=opt_state,
                    devices=devices,
                    weight_transformer_dropout=cfg.training.weight_transformer_dropout,
                    prop_cells_updated=cfg.training.prop_cells_updated,
                    grad_clip_norm=cfg.training.grad_clip_norm,
                    n_spatial_dims=n_spatial_dims,
                )
            else:
                (
                    new_tasknet_data,
                    batch_avg_loss,
                    local_rule_params,
                    opt_state,
                    rand_key,
                ) = metanca_train_step(
                    batch=batch,
                    local_rule_net_apply=training_vars.local_rule_net_apply,
                    local_rule_params=local_rule_params,
                    rand_key=rand_key,
                    tasknet_data_list=tasknet_data_list,
                    adjs=adjs,
                    tasknet_param_names=tasknet_param_names,
                    tasknet_apply_fns=tasknet_apply_fns,
                    n_update_steps=n_update_steps,
                    hidden_dim=hidden_dim,
                    optimizer=optimizer,
                    optimizer_state=opt_state,
                    weight_transformer_dropout=cfg.training.weight_transformer_dropout,
                    prop_cells_updated=cfg.training.prop_cells_updated,
                    grad_clip_norm=cfg.training.grad_clip_norm,
                    n_spatial_dims=n_spatial_dims,
                )
            if timing_enabled:
                batch_avg_loss = jax.block_until_ready(batch_avg_loss)
            t_after_train = time.time()

            # NaN / loss spike detection: break and roll back entire metaepoch
            is_nan = not jnp.isfinite(batch_avg_loss)
            is_spike = (
                ema_loss is not None
                and spike_multiplier > 0
                and batch_avg_loss > spike_multiplier * ema_loss
            )
            if is_nan or is_spike:
                reason = (
                    "NaN/inf"
                    if is_nan
                    else f"spike ({batch_avg_loss:.4f} > {spike_multiplier}×EMA {ema_loss:.4f})"
                )
                logger.warning(
                    f"{reason} loss at metaepoch {metaepoch} batch {i}, "
                    "rolling back entire metaepoch"
                )
                if is_spike:
                    loss_spike_rollbacks += 1
                nan_detected = True
                break

            # Update sample pools with post-update tasknet states
            if pools:
                for arch_idx, (new_params, new_hs, _pe) in enumerate(new_tasknet_data):
                    pools[arch_idx].add(
                        SamplePool.to_numpy(new_params),
                        SamplePool.to_numpy(new_hs),
                    )

            t_callback_start = time.time()
            ctx = ctx.with_updated(
                batch_idx=batch_idx,
                tasknet_data_list=new_tasknet_data,
                batch_x=batch_x,
                batch_y=batch_y,
                mask=mask,
            )
            callback_runner, batch_result = callback_runner.on_batch_end(ctx)
            t_after_callback = time.time()

            metrics = dict(batch_result.logs)
            metrics["elapsed_time"] = time.time() - t_start_batch
            metrics["loss"] = batch_avg_loss
            if timing_enabled:
                metrics["data_prep_time"] = t_after_prep - t_start_batch
                metrics["train_step_time"] = t_after_train - t_train_start
                metrics["callback_time"] = t_after_callback - t_callback_start

            for metric in metrics:
                metric_totals[metric] = metric_totals.get(metric, 0.0) + metrics[metric]
                if metric not in batch_metrics:
                    batch_metrics[metric] = {
                        "sum": jnp.array(0.0),
                        "count": jnp.array(0, dtype=jnp.int32),
                    }
                batch_metrics[metric]["sum"] = batch_metrics[metric]["sum"] + metrics[metric]
                batch_metrics[metric]["count"] = batch_metrics[metric]["count"] + 1

            if i > 0 and (i % cfg.logging.log_every_n_batches == 0):
                logger.debug(f"metaepoch {metaepoch} | batch {i}/{n_train_batches}")
                logger.debug(f"\tMetrics (last {cfg.logging.log_every_n_batches} batches)")
                for metric_name, metric_batch in batch_metrics.items():
                    count = metric_batch["count"]
                    mean = metric_batch["sum"] / jnp.maximum(count, 1)
                    logger.debug(f"\t> {metric_name}={float(mean):0.4f}")
                    batch_metrics[metric_name]["sum"] = jnp.array(0.0)
                    batch_metrics[metric_name]["count"] = jnp.array(0, dtype=jnp.int32)

        # NaN/spike rollback: restore pre-metaepoch state and skip this metaepoch
        if nan_detected:
            local_rule_params = metaepoch_local_rule_params
            opt_state = metaepoch_opt_state
            rand_key = metaepoch_rand_key
            if use_multi_gpu:
                lr_params_per_device = metaepoch_lr_params_per_device
            consecutive_rollbacks += 1
            if pools and consecutive_rollbacks >= max_consecutive_rollbacks:
                for pool in pools:
                    pool.clear()
                pool_flushes += 1
                consecutive_rollbacks = 0
                logger.warning(
                    f"Flushed all sample pools after {max_consecutive_rollbacks} "
                    f"consecutive rollbacks at metaepoch {metaepoch} "
                    f"(total flushes: {pool_flushes})"
                )
            continue

        consecutive_rollbacks = 0

        metric_avgs = {metric: total / n_train_batches for metric, total in metric_totals.items()}

        # Update EMA loss after clean metaepoch
        if ema_loss is None:
            ema_loss = float(metric_avgs["loss"])
        else:
            ema_loss = 0.01 * float(metric_avgs["loss"]) + 0.99 * ema_loss

        pool_metrics = {}
        if pools:
            pool_metrics["pool/avg_size"] = sum(p.size for p in pools) / len(pools)
        spike_metrics = {}
        if spike_multiplier > 0:
            spike_metrics["ema_loss"] = ema_loss
            spike_metrics["loss_spike_rollbacks"] = loss_spike_rollbacks
            spike_metrics["pool_flushes"] = pool_flushes
        best_metrics = log_dict(
            {f"train_archs/train_data_{name}": value for name, value in metric_avgs.items()}
            | pool_metrics
            | spike_metrics
            | {"metaepoch": metaepoch, "n_update_steps": n_update_steps},
            step=metaepoch,
            prefix="Training",
            best_metrics=best_metrics,
        )

        save_checkpoint(
            metaepoch,
            metric_avgs,
            local_rule_params,
            opt_state,
            training_vars.checkpoint_metadata,
            training_vars.checkpoint_manager,
        )

        elapsed_times = elapsed_times.at[metaepoch % cfg.logging.log_every_n_metaepochs].set(
            time.time() - t_start
        )
        if metaepoch % cfg.logging.log_every_n_metaepochs == 0:
            mean_elapsed = elapsed_times.mean()
            loss = metric_avgs["loss"]

            logger.info(f"metaepoch: {metaepoch}: local_rule_loss={loss:0.4f}")
            logger.info(
                f"Avg elapsed time (last {cfg.logging.log_every_n_metaepochs}): "
                f"{mean_elapsed:0.3f}"
            )
            elapsed_times = elapsed_times.at[:].set(0.0)
            # Run validation
            rand_key, val_key = jax.random.split(rand_key)
            val_metrics, callback_runner = metanca_validation_step(
                val_batches=val_batches,
                local_rule_net_apply=training_vars.local_rule_net_apply,
                local_rule_params=local_rule_params,
                rand_key=val_key,
                val_tasknet=training_vars.test_tasknet,
                n_update_steps=n_update_steps,
                hidden_dim=hidden_dim,
                prop_cells_updated=cfg.training.prop_cells_updated,
                weight_transformer_dropout=cfg.training.weight_transformer_dropout,
                n_spatial_dims=n_spatial_dims,
                callback_runner=callback_runner,
            )

            best_metrics = log_dict(
                {f"val_arch/val_data_{name}": value for name, value in val_metrics.items()},
                step=metaepoch,
                prefix="Validation (val arch, val data)",
                best_metrics=best_metrics,
            )

            # Run validation with 2x update steps
            rand_key, val_2x_key = jax.random.split(rand_key)
            val_2x_metrics, callback_runner = metanca_validation_step(
                val_batches=val_batches,
                local_rule_net_apply=training_vars.local_rule_net_apply,
                local_rule_params=local_rule_params,
                rand_key=val_2x_key,
                val_tasknet=training_vars.test_tasknet,
                n_update_steps=n_update_steps * 2,
                hidden_dim=hidden_dim,
                prop_cells_updated=cfg.training.prop_cells_updated,
                weight_transformer_dropout=cfg.training.weight_transformer_dropout,
                n_spatial_dims=n_spatial_dims,
                callback_runner=callback_runner,
            )

            best_metrics = log_dict(
                {f"val_arch/val_data_2x_{name}": value for name, value in val_2x_metrics.items()},
                step=metaepoch,
                prefix="Validation 2x (val arch, val data)",
                best_metrics=best_metrics,
            )

            # Run validation on each training architecture, average metrics
            train_arch_metric_totals = {}
            for arch_idx, train_tasknet in enumerate(tasknets):
                rand_key, train_val_key = jax.random.split(rand_key)
                train_arch_val_metrics, callback_runner = metanca_validation_step(
                    val_batches=val_batches,
                    local_rule_net_apply=training_vars.local_rule_net_apply,
                    local_rule_params=local_rule_params,
                    rand_key=train_val_key,
                    val_tasknet=train_tasknet,
                    n_update_steps=n_update_steps,
                    hidden_dim=hidden_dim,
                    prop_cells_updated=cfg.training.prop_cells_updated,
                    weight_transformer_dropout=cfg.training.weight_transformer_dropout,
                    n_spatial_dims=n_spatial_dims,
                    callback_runner=callback_runner,
                )
                for name, value in train_arch_val_metrics.items():
                    train_arch_metric_totals[name] = train_arch_metric_totals.get(name, 0.0) + value
            n_train_archs = len(tasknets)
            train_arch_val_avgs = {
                name: total / n_train_archs for name, total in train_arch_metric_totals.items()
            }
            best_metrics = log_dict(
                {
                    f"train_archs/val_data_{name}": value
                    for name, value in train_arch_val_avgs.items()
                },
                step=metaepoch,
                prefix="Validation (train archs, val data)",
                best_metrics=best_metrics,
            )

            # Run stateful callbacks after validation
            ctx = ctx.with_updated(
                metaepoch=metaepoch,
                metrics={**metric_avgs, **val_metrics},
                local_rule_params=local_rule_params,
                optimizer_state=opt_state,
            )
            callback_runner, result = callback_runner.on_validation_end(ctx)

            if result.logs:
                wandb.log(result.logs, step=metaepoch)

            if result.stop_training:
                logger.info("Early stopping triggered, ending training")
                break

    # Finalize callbacks
    callback_runner, _ = callback_runner.on_train_end(ctx)
