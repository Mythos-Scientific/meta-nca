from typing import Callable, Literal

import jax


def _identity(x: jax.Array) -> jax.Array:
    return x


_act2callable = {
    "leaky_relu": jax.nn.leaky_relu,
    "silu": jax.nn.silu,
    "relu": jax.nn.relu,
    "identity": _identity,
    "elu": jax.nn.elu,
    "tanh": jax.nn.tanh,
}


def get_activation(
    act: Literal["leaky_relu", "silu", "relu", "identity", "elu", "tanh"] | None,
) -> Callable[[jax.Array], jax.Array]:
    return _act2callable[act or "identity"]
