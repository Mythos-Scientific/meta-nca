from typing import Optional

import jax


def swap_axes_and_reshape(
    arr: jax.Array, index_dim: int, feature_dim_size: int, stride: Optional[int] = None
) -> jax.Array:
    if stride is None:
        index_dim_size = arr.shape[index_dim]
        return arr.swapaxes(0, index_dim).reshape(index_dim_size, -1, feature_dim_size)
    else:
        index_dim_size = arr.shape[index_dim] // stride
        return (
            arr.reshape(index_dim_size, stride, -1, feature_dim_size)
            .swapaxes(0, 1)
            .reshape(stride, -1, feature_dim_size)
        )


def invert_swap_axes_and_reshape(
    arr: jax.Array, original_shape: tuple[int, ...], index_dim: int, stride: Optional[int] = None
) -> jax.Array:
    # First reshape back to the swapped shape
    if stride is None:
        swapped_shape = list(original_shape)
        swapped_shape[0], swapped_shape[index_dim] = (
            swapped_shape[index_dim],
            swapped_shape[0],
        )

        reshaped = arr.reshape(swapped_shape)
        # Then swap the axes back
        return reshaped.swapaxes(0, index_dim)
    else:
        full_index_dim_size = original_shape[index_dim]
        index_dim_size = full_index_dim_size // stride
        return (
            arr.reshape(stride, index_dim_size, -1, arr.shape[-1])
            .swapaxes(1, 0)
            .reshape(original_shape)
        )
        # return NotImplemented
        # index_dim_size = original_shape[index_dim]
        # arr.swapaxes(1, 0).reshape(index_dim_size, stride, -1, feature_dim_size)
