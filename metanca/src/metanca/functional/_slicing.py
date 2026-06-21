import jax


def slice_0th_axis(
    start_index: int, leading_ax_param: jax.Array, slice_size: int, stride: int
) -> jax.Array:
    chunk = jax.lax.dynamic_slice_in_dim(
        leading_ax_param, start_index=start_index, slice_size=slice_size, axis=0
    )
    return chunk[::stride, ...]


batched_slice = jax.jit(
    jax.vmap(slice_0th_axis, in_axes=(0, None, None, None)), static_argnums=(2, 3)
)
