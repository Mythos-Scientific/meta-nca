import jax
import jax.numpy as jnp


def get_weight_mask(
    key: jax.random.PRNGKey,
    param: jax.Array,
    proportion: float = 0.8,
) -> jax.Array:
    key, subkey = jax.random.split(key)

    num_cells_updated = int(proportion * param.size)
    flat_chosen_inds = jax.random.choice(subkey, param.size, (num_cells_updated,), replace=False)
    mask = jnp.zeros(param.size, dtype=bool).at[flat_chosen_inds].set(True).reshape(param.shape)
    return mask
