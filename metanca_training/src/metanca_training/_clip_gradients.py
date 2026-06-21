import chex
import jax
import jax.numpy as jnp


def _sum_of_squares(x: chex.Array) -> chex.Scalar:
    return jnp.sum(jnp.square(x))


def clip_gradients(gradients: chex.ArrayTree, grad_clip_norm: float = 1.0) -> chex.ArrayTree:
    # Gradient clipping and parameter update
    total_norm = jnp.sqrt(
        jnp.sum(
            jnp.asarray(
                jax.tree_util.tree_leaves(jax.tree_util.tree_map(_sum_of_squares, gradients))
            )
        )
    )
    clip_coef = jnp.minimum(1.0, grad_clip_norm / (total_norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * clip_coef, gradients)
