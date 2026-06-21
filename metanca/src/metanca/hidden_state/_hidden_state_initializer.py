import itertools
from typing import Optional, Sequence

import jax
import jax.numpy as jnp

from metanca.typing import HiddenStateInitializer

from ._positional_encoding import positional_encoding


def hidden_state_initializer(
    maximum_dims: jax.Array, n_layers: int, d_neuron: int = 2, d_spatial: int = 2, d_layer: int = 2
) -> HiddenStateInitializer:
    """
    Arguments
    ---------
    maximum_dims: jax.Array
        Maximum dimensionalities along stacked/aligned parameter shapes.
        The assumed order of the maximum dim array is: (*spatial_dims, max_in_dim, max_out_dim)
    n_layers: int
        Number of layers in the task network
    d_encoding: int | Sequence[int]
        The width of positional encodings for each positional encoding type.
        If passing a Sequence[int], there should be 1 integer per positional encoding buffer.
        The number of positional encoding buffers is:
            # spatial dimensions + 1 (neuron encoding) + 1 (layer encoding)

        E.g., for images `d_encoding` could be [10, 10, 10,                      10]
                                                ^ h ^ w ^ max(in_dim, out_dim)    ^ layers

    Returns
    -------
    HiddenStateInitializer
    """
    *max_spatial_dims, max_in_dim, max_out_dim = maximum_dims
    *_, in_dim_ind, out_dim_ind = range(len(maximum_dims))

    n_spatial_dims = len(max_spatial_dims)

    encoding_widths = ((d_spatial,) * n_spatial_dims) + (d_neuron, d_layer)

    dims = (*max_spatial_dims, max(max_in_dim, max_out_dim), n_layers)
    *spatial_buffers, neuron_buffer, layer_buffer = [
        positional_encoding(dim, d_model=width) for dim, width in zip(dims, encoding_widths)
    ]

    # Bias sentinel PE: valid sin/cos at one-past-max position for each buffer.
    # Uses a real PE value so RoPE doesn't amplify, but at a position no kernel
    # occupies so the local rule can distinguish bias from kernel.
    bias_spatial_sentinels = [
        positional_encoding(dim + 1, d_model=d_spatial)[-1] for dim in max_spatial_dims
    ]
    bias_neuron_sentinel = positional_encoding(max(max_in_dim, max_out_dim) + 1, d_model=d_neuron)[
        -1
    ]

    hidden_channel_widths = ([d_spatial] * n_spatial_dims) + ([d_neuron] * 2) + [d_layer]
    all_channel_indices = list(itertools.accumulate(hidden_channel_widths))
    total_hidden_channels = sum(hidden_channel_widths)

    # DEBUG: Verify all channel widths are equal (required by split_into_parts in rotary attention)
    assert all(w == hidden_channel_widths[0] for w in hidden_channel_widths), (
        f"split_into_parts assumes equal channel widths but got {hidden_channel_widths}. "
        f"Ensure d_spatial={d_spatial}, d_neuron={d_neuron}, d_layer={d_layer} are all equal."
    )

    def _get_buffers_and_channel_dims(
        kernel_shape: Sequence[int],
    ) -> tuple[list[jax.Array], list[int], list[tuple[int, int]]]:
        if len(kernel_shape) != len(maximum_dims):
            buffer_list = [neuron_buffer, neuron_buffer, layer_buffer]
            start_indices = all_channel_indices[n_spatial_dims - 1 : -1]
            channel_width_list = hidden_channel_widths[n_spatial_dims:]
        else:
            buffer_list = spatial_buffers + [neuron_buffer, neuron_buffer, layer_buffer]
            start_indices = [0] + all_channel_indices[:-1]
            channel_width_list = hidden_channel_widths

        end_indices = start_indices[1:] + [total_hidden_channels]
        return buffer_list, channel_width_list, list(zip(start_indices, end_indices))

    def initializer(
        kernel_shape: Sequence[int],
        layer_idx: int,
        bias: bool = True,
        bias_constant: float = -10.0,
    ) -> tuple[jax.Array, Optional[jax.Array]]:

        # DEBUG: Verify layer_idx is within bounds (JAX silently clamps out-of-bounds indices)
        assert 0 <= layer_idx < n_layers, (
            f"layer_idx={layer_idx} out of bounds for n_layers={n_layers}. "
            f"This would cause silent index clamping in JAX!"
        )

        kernel_hidden_state = jnp.zeros((*kernel_shape, total_hidden_channels))
        rank = len(kernel_shape)

        buffer_list, widths, index_intervals = _get_buffers_and_channel_dims(kernel_shape)

        # target_shape = kernel_shape + (total_hidden_channels,)

        iterator = zip(
            kernel_shape + (layer_idx,),
            buffer_list,
            widths,
            index_intervals,
        )

        for axis, (kernel_dim, buffer, width, (start_idx, end_idx)) in enumerate(iterator):
            if axis < len(kernel_shape):
                # DEBUG: Verify kernel_dim doesn't exceed buffer size (JAX silently clamps)
                assert kernel_dim <= buffer.shape[0], (
                    f"axis={axis}: kernel_dim={kernel_dim} exceeds buffer size={buffer.shape[0]}. "
                    f"This would cause silent index clamping in JAX!"
                )
                expanded_shape = ((1,) * axis) + (kernel_dim,) + (1,) * (rank - 1 - axis) + (width,)
                update = jnp.broadcast_to(
                    buffer[:kernel_dim].reshape(expanded_shape), kernel_shape + (width,)
                )
            else:
                # DEBUG: Verify layer index doesn't exceed buffer size
                assert kernel_dim < buffer.shape[0], (
                    f"layer axis: layer_idx={kernel_dim} exceeds layer_buffer size={buffer.shape[0]}. "
                    f"This would cause silent index clamping in JAX!"
                )
                expanded_shape = ((1,) * rank) + (width,)
                update = jnp.broadcast_to(
                    buffer[kernel_dim].reshape(expanded_shape), kernel_shape + (width,)
                )

            kernel_hidden_state = jax.lax.dynamic_update_slice(
                kernel_hidden_state, update, start_indices=[0] * rank + [start_idx]
            )

        # For Dense params (fewer dims than Conv), fill spatial PE channels with
        # PE(0) instead of leaving them as zeros. This ensures Dense params have
        # a valid spatial encoding for RoPE attention heads.
        if len(kernel_shape) != len(maximum_dims):
            for s_idx in range(n_spatial_dims):
                buf = spatial_buffers[s_idx]
                pe_zero = buf[0]  # PE(0) = [0, 1, 0, 1, ...], shape: (d_spatial,)
                start = 0 if s_idx == 0 else all_channel_indices[s_idx - 1]
                end = all_channel_indices[s_idx]
                kernel_hidden_state = kernel_hidden_state.at[..., start:end].set(pe_zero)

        if bias:
            slice_start_indices = [0] * len(kernel_hidden_state.shape)
            slice_sizes = (1,) * (rank - 1) + (
                kernel_shape[-1],
                total_hidden_channels,
            )

            bias_hidden_state = jnp.squeeze(
                jax.lax.dynamic_slice(
                    kernel_hidden_state,
                    start_indices=slice_start_indices,
                    slice_sizes=slice_sizes,
                ),
            )

            # Overwrite spatial and in_neuron PE channels with sentinel PE values
            # (PE at max_dim+1). These are valid sin/cos values so RoPE doesn't
            # amplify, but at positions no kernel occupies so the local rule can
            # distinguish bias from kernel parameters.
            for s_idx in range(n_spatial_dims):
                start = 0 if s_idx == 0 else all_channel_indices[s_idx - 1]
                end = all_channel_indices[s_idx]
                bias_hidden_state = bias_hidden_state.at[:, start:end].set(
                    bias_spatial_sentinels[s_idx]
                )

            # In_neuron sentinel
            in_neuron_start = all_channel_indices[n_spatial_dims - 1] if n_spatial_dims > 0 else 0
            in_neuron_end = all_channel_indices[n_spatial_dims]
            bias_hidden_state = bias_hidden_state.at[:, in_neuron_start:in_neuron_end].set(
                bias_neuron_sentinel
            )
        else:
            bias_hidden_state = None
        return kernel_hidden_state, bias_hidden_state

    return initializer, total_hidden_channels
