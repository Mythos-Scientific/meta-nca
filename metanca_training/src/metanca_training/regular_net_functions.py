from typing import Any, Optional

import chex
import jax
import jax.numpy as jnp
import optax
from evosax.algorithms import SimpleES
from flax.training import train_state
from jax import jit, nn

from metanca.nn import ConvMLP, MultiLayerPerceptron, ResNet
from metanca_training._loss import compute_loss
from metanca_training.callbacks._accuracy import accuracy


class RegularNetTrainState(train_state.TrainState):
    batch_stats: Optional[chex.ArrayTree] = None


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _build_regular_model(arch, no_bias: bool):
    if isinstance(arch, list):
        if len(arch) != 1:
            raise ValueError("Architecture list must contain exactly one architecture definition.")
        arch = arch[0]

    stage_blocks = _cfg_get(arch, "stage_blocks")
    num_classes = _cfg_get(arch, "num_classes")
    if stage_blocks is not None and num_classes is not None:
        use_bn = bool(_cfg_get(arch, "use_bn", False))
        return ResNet(
            num_classes=int(num_classes),
            stage_blocks=tuple(stage_blocks),
            base_channels=int(_cfg_get(arch, "base_channels", 16)),
            use_bn=use_bn,
            act=str(_cfg_get(arch, "act", "relu")),
        )

    conv_layers, dense_layer_widths = arch
    use_bias = not no_bias
    if conv_layers:
        return ConvMLP(
            conv_layer_specs=conv_layers,
            mlp_layer_specs=dense_layer_widths,
            conv_use_bias=use_bias,
            mlp_use_bias=use_bias,
        )
    return MultiLayerPerceptron(layer_specs=dense_layer_widths, use_bias=use_bias)


def _model_uses_batch_stats(model: Any) -> bool:
    return bool(getattr(model, "use_bn", False))


def _init_model_state(model, rand_key: jax.Array, sample_x: jax.Array):
    sample_x = _maybe_to_nhwc(sample_x)
    if _model_uses_batch_stats(model):
        variables = model.init(rand_key, sample_x, training=True)
        return variables["params"], variables["batch_stats"]
    variables = model.init(rand_key, sample_x)
    return variables["params"], None


def _pack_model_state(params: Any, batch_stats: Any):
    if batch_stats is None:
        return params
    return {"params": params, "batch_stats": batch_stats}


def _unpack_model_state(model_state: Any):
    if isinstance(model_state, dict) and "params" in model_state and "batch_stats" in model_state:
        return model_state["params"], model_state["batch_stats"]
    return model_state, None


def _model_apply(
    model: Any,
    params: Any,
    x: jax.Array,
    *,
    batch_stats: Any = None,
    training: bool = False,
):
    x = _maybe_to_nhwc(x)
    if batch_stats is not None:
        return model.apply({"params": params, "batch_stats": batch_stats}, x, training=training)
    if _model_uses_batch_stats(model):
        raise ValueError(
            "BatchNorm model requires batch_stats in model state. "
            "Initialize or restore regular-net state with batch_stats."
        )
    return model.apply({"params": params}, x)


def _loss_fn_factory(model):
    def apply_fn(params, x):
        unpacked_params, unpacked_batch_stats = _unpack_model_state(params)
        return _model_apply(
            model,
            unpacked_params,
            x,
            batch_stats=unpacked_batch_stats,
            training=False,
        )

    def loss_fn(params, x, y):
        if isinstance(y, tuple):
            labels, mask = y
        else:
            labels = y
            mask = jnp.ones((x.shape[0], 1), dtype=jnp.bool_)
        return compute_loss(_maybe_to_nhwc(x), labels, mask, params, apply_fn)

    return loss_fn


def _maybe_to_nhwc(x: jax.Array) -> jax.Array:
    if x.ndim == 4 and x.shape[1] in (1, 3) and x.shape[-1] not in (1, 3):
        return jnp.transpose(x, (0, 2, 3, 1))
    return x


def update_params(optimizer, loss_fn):
    """
    Generic function for optimization of params based on supplied optimizer and loss_fn.
    """

    def update_params_inner(params, opt_state, x, y):
        grads = jax.grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    return update_params_inner


def _sample_input_from_batches(train_X: Any) -> jax.Array:
    if isinstance(train_X, (list, tuple)):
        sample = train_X[0]
    else:
        sample = train_X
    return sample


def _extract_labels_and_mask(x: jax.Array, y: Any) -> tuple[jax.Array, jax.Array]:
    if isinstance(y, tuple):
        return y
    return y, jnp.ones((x.shape[0], 1), dtype=jnp.bool_)


def _batch_loss_from_logits(logits: jax.Array, labels: jax.Array, mask: jax.Array) -> jax.Array:
    valid_count = jnp.sum(mask)
    return jnp.sum(optax.softmax_cross_entropy(logits, labels) * mask.squeeze(-1)) / jnp.maximum(
        valid_count, 1
    )


def init_and_train_regular_network(
    rand_key,
    layer_widths,
    train_X,
    train_y,
    val_X,
    val_y,
    no_bias=False,
    epochs=10000,
    params=None,
    optimizer=None,
    opt_state=None,
    model=None,
):
    rand_key, reg_net_rand_key = jax.random.split(rand_key, 2)
    model = model or _build_regular_model(layer_widths, no_bias=no_bias)
    if params is None:
        sample_x = _sample_input_from_batches(train_X)
        params, batch_stats = _init_model_state(model, reg_net_rand_key, sample_x)
    else:
        params, batch_stats = _unpack_model_state(params)
        if _model_uses_batch_stats(model) and batch_stats is None:
            sample_x = _sample_input_from_batches(train_X)
            _, batch_stats = _init_model_state(model, reg_net_rand_key, sample_x)

    if optimizer is None:
        optimizer = optax.adam(learning_rate=0.001)
    if opt_state is None:
        opt_state = optimizer.init(params)

    state = RegularNetTrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        opt_state=opt_state,
        batch_stats=batch_stats,
    )

    all_train_X = jnp.concatenate(train_X, axis=0)
    all_train_y = jnp.concatenate(train_y, axis=0)
    all_val_X = jnp.concatenate(val_X, axis=0)
    all_val_y = jnp.concatenate(val_y, axis=0)

    def compute_batch_loss(params, batch_stats, x, y):
        labels, mask = _extract_labels_and_mask(x, y)
        logits = _model_apply(model, params, x, batch_stats=batch_stats, training=False)
        return _batch_loss_from_logits(logits, labels, mask)

    if state.batch_stats is not None:

        @jit
        def train_step(train_state, x, y):
            labels, mask = _extract_labels_and_mask(x, y)

            def loss_fn(current_params):
                logits, updates = model.apply(
                    {"params": current_params, "batch_stats": train_state.batch_stats},
                    _maybe_to_nhwc(x),
                    training=True,
                    mutable=["batch_stats"],
                )
                loss = _batch_loss_from_logits(logits, labels, mask)
                return loss, updates["batch_stats"]

            (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                train_state.params
            )
            train_state = train_state.apply_gradients(grads=grads)
            train_state = train_state.replace(batch_stats=new_batch_stats)
            return train_state, loss

    else:

        @jit
        def train_step(train_state, x, y):
            labels, mask = _extract_labels_and_mask(x, y)

            def loss_fn(current_params):
                logits = _model_apply(model, current_params, x, training=True)
                return _batch_loss_from_logits(logits, labels, mask)

            loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

    reg_net_params_history = [_pack_model_state(state.params, state.batch_stats)]
    for i in range(epochs):
        epoch_train_loss = 0
        for batch_x, batch_y in zip(train_X, train_y):
            state, batch_loss = train_step(state, batch_x, batch_y)
            epoch_train_loss += batch_loss

        reg_net_params_history.append(_pack_model_state(state.params, state.batch_stats))

        val_loss = compute_batch_loss(state.params, state.batch_stats, all_val_X, all_val_y)
        train_loss = compute_batch_loss(state.params, state.batch_stats, all_train_X, all_train_y)
        val_acc = evaluate_reg_net(
            model,
            _pack_model_state(state.params, state.batch_stats),
            all_val_X,
            all_val_y,
        )
        print(f"Train loss at step {i}: {train_loss}")
        print(f"Val loss: {val_loss}")
        print(f"Val accuracy: {val_acc}")

    return (
        rand_key,
        _pack_model_state(state.params, state.batch_stats),
        reg_net_params_history,
        model,
        optimizer,
        state.opt_state,
    )


def init_and_train_regular_network_batched(
    rand_key,
    layer_widths,
    train_batches_per_device,
    val_batches_per_device,
    no_bias=False,
    epochs=10000,
    params=None,
    optimizer=None,
    opt_state=None,
    model=None,
):
    rand_key, reg_net_rand_key = jax.random.split(rand_key, 2)
    model = model or _build_regular_model(layer_widths, no_bias=no_bias)

    if params is None:
        sample_x = train_batches_per_device[0][0]
        params, batch_stats = _init_model_state(model, reg_net_rand_key, sample_x)
    else:
        params, batch_stats = _unpack_model_state(params)
        if _model_uses_batch_stats(model) and batch_stats is None:
            sample_x = train_batches_per_device[0][0]
            _, batch_stats = _init_model_state(model, reg_net_rand_key, sample_x)

    if optimizer is None:
        optimizer = optax.adam(learning_rate=0.001)
    if opt_state is None:
        opt_state = optimizer.init(params)

    state = RegularNetTrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        opt_state=opt_state,
        batch_stats=batch_stats,
    )

    def compute_batch_loss(params, batch_stats, x, y):
        labels, mask = _extract_labels_and_mask(x, y)
        logits = _model_apply(model, params, x, batch_stats=batch_stats, training=False)
        return _batch_loss_from_logits(logits, labels, mask)

    if state.batch_stats is not None:

        @jit
        def train_step(train_state, x, y):
            labels, mask = _extract_labels_and_mask(x, y)

            def loss_fn(current_params):
                logits, updates = model.apply(
                    {"params": current_params, "batch_stats": train_state.batch_stats},
                    _maybe_to_nhwc(x),
                    training=True,
                    mutable=["batch_stats"],
                )
                loss = _batch_loss_from_logits(logits, labels, mask)
                return loss, updates["batch_stats"]

            (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                train_state.params
            )
            train_state = train_state.apply_gradients(grads=grads)
            train_state = train_state.replace(batch_stats=new_batch_stats)
            return train_state, loss

    else:

        @jit
        def train_step(train_state, x, y):
            labels, mask = _extract_labels_and_mask(x, y)

            def loss_fn(current_params):
                logits = _model_apply(model, current_params, x, training=True)
                return _batch_loss_from_logits(logits, labels, mask)

            loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

    reg_net_params_history = [_pack_model_state(state.params, state.batch_stats)]
    for i in range(epochs):
        epoch_train_loss = 0
        for batch_x, batch_y, batch_mask in zip(
            train_batches_per_device[0],
            train_batches_per_device[1],
            train_batches_per_device[2],
        ):
            state, batch_loss = train_step(state, batch_x, (batch_y, batch_mask))
            epoch_train_loss += batch_loss

        epoch_train_loss /= len(train_batches_per_device[0])

        reg_net_params_history.append(_pack_model_state(state.params, state.batch_stats))

        epoch_val_loss = 0
        for batch_x, batch_y, batch_mask in zip(
            val_batches_per_device[0],
            val_batches_per_device[1],
            val_batches_per_device[2],
        ):
            epoch_val_loss += compute_batch_loss(
                state.params, state.batch_stats, batch_x, (batch_y, batch_mask)
            )

        epoch_val_loss /= len(val_batches_per_device[0])

        epoch_val_acc = evaluate_reg_net_batched(
            model,
            _pack_model_state(state.params, state.batch_stats),
            val_batches_per_device,
        )

        print(f"Train loss at step {i}: {epoch_train_loss}")
        print(f"Val loss: {epoch_val_loss}")
        print(f"Val accuracy: {epoch_val_acc}")

    return (
        rand_key,
        _pack_model_state(state.params, state.batch_stats),
        reg_net_params_history,
        model,
        optimizer,
        state.opt_state,
    )


def evolutionary_strategies_batched(
    rand_key,
    reg_net_arch,
    train_batches_per_device,
    val_batches_per_device,
    no_bias=False,
    num_generations=100,
    pop_size=32,
):
    rand_key, reg_net_rand_key = jax.random.split(rand_key, 2)
    model = _build_regular_model(reg_net_arch, no_bias=no_bias)
    sample_x = train_batches_per_device[0][0]
    dummy_solution, _ = _init_model_state(model, reg_net_rand_key, sample_x)
    num_batches = train_batches_per_device[0].shape[0]

    loss_fn = _loss_fn_factory(model)

    population_loss_fn = jax.vmap(
        loss_fn, (0, None, None)
    )  # i think 0 should cover all the parameters in the pytre
    es = SimpleES(population_size=pop_size, solution=dummy_solution)
    es_params = es.default_params

    # Initialize state
    key = jax.random.key(0)
    state = es.init(key, dummy_solution, es_params)

    # Ask-Eval-Tell loop
    for i in range(num_generations):
        key, key_ask, key_eval, key_tell = jax.random.split(key, 4)

        # Generate a set of candidate solutions to evaluate
        population, state = es.ask(key_ask, state, es_params)

        loss = 0
        for batch_x, batch_y, batch_mask in zip(
            train_batches_per_device[0],
            train_batches_per_device[1],
            train_batches_per_device[2],
        ):
            loss += population_loss_fn(population, batch_x, (batch_y, batch_mask))

        # Evaluate the fitness of the population
        fitness = loss / num_batches

        # Update the evolution strategy
        state, metrics = es.tell(key_tell, population, fitness, state, es_params)
        if i % 100 == 0:
            print("Generation " + str(i))
            print("Best loss curr gen: " + str(loss.min()))

    best_net = es._unravel_solution(state.best_solution)
    reg_net_train_accuracy = evaluate_reg_net_batched(
        model, best_net, train_batches_per_device, no_bias=no_bias
    )
    reg_net_val_accuracy = evaluate_reg_net_batched(
        model, best_net, val_batches_per_device, no_bias=no_bias
    )
    return state.best_solution, state.best_fitness, reg_net_train_accuracy, reg_net_val_accuracy


def evaluate_reg_net(model, params, X_val, y_val, no_bias=False):
    unpacked_params, batch_stats = _unpack_model_state(params)
    logits = _model_apply(model, unpacked_params, X_val, batch_stats=batch_stats, training=False)
    y_pred = nn.softmax(logits)
    return accuracy(y_val, y_pred)


def evaluate_reg_net_batched(model, params, batches_per_device, no_bias=False):
    unpacked_params, batch_stats = _unpack_model_state(params)
    acc = 0

    for batch_x, batch_y, batch_mask in zip(
        batches_per_device[0],
        batches_per_device[1],
        batches_per_device[2],
    ):
        logits = _model_apply(
            model,
            unpacked_params,
            batch_x,
            batch_stats=batch_stats,
            training=False,
        )
        y_pred = nn.softmax(logits, axis=-1)
        acc += accuracy(batch_y, y_pred, mask=batch_mask)
    acc /= len(batches_per_device[0])
    return acc


def regular_net_baseline(
    rand_key, arch, train_X, train_y, val_X, val_y, no_bias=False, epochs=10000
):
    (
        rand_key,
        reg_net_params,
        reg_net_params_history,
        model,
        _optimizer,
        _opt_state,
    ) = init_and_train_regular_network(
        rand_key, arch, train_X, train_y, val_X, val_y, no_bias=no_bias, epochs=epochs
    )
    all_val_X = jnp.concatenate(val_X)
    all_val_y = jnp.concatenate(val_y)
    reg_net_val_accuracy = evaluate_reg_net(
        model, reg_net_params, all_val_X, all_val_y, no_bias=no_bias
    )
    return reg_net_params_history, reg_net_val_accuracy


def regular_net_baseline_batched(
    rand_key, arch, train_batches_per_device, val_batches_per_device, no_bias=False, epochs=10000
):
    (
        rand_key,
        reg_net_params,
        reg_net_params_history,
        model,
        _optimizer,
        _opt_state,
    ) = init_and_train_regular_network_batched(
        rand_key,
        arch,
        train_batches_per_device,
        val_batches_per_device,
        no_bias=no_bias,
        epochs=epochs,
    )
    reg_net_train_accuracy = evaluate_reg_net_batched(
        model, reg_net_params, train_batches_per_device, no_bias=no_bias
    )
    reg_net_val_accuracy = evaluate_reg_net_batched(
        model, reg_net_params, val_batches_per_device, no_bias=no_bias
    )
    return reg_net_params_history, reg_net_train_accuracy, reg_net_val_accuracy
