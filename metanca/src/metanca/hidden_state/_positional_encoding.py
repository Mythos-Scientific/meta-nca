import jax
import jax.numpy as jnp


def positional_encoding(max_position: int, d_model: int) -> jax.Array:
    pos_enc = jnp.zeros((max_position, d_model))
    position = jnp.arange(0, max_position)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))

    pos_enc = pos_enc.at[:, 0::2].set(jnp.sin(position * div_term))
    pos_enc = pos_enc.at[:, 1::2].set(jnp.cos(position * div_term))
    return pos_enc
