import chex
import jax
import jax.numpy as jnp
from jax import random, vmap

from metanca.hidden_state import positional_encoding
from metanca.nn import LocalRuleNet

"""
init MetaNCA functions:
"""


def reset_task_nets(
    archs, task_net_rand_keys, d_model_layer, d_model_neuron, d_model_position, no_bias=False
):
    return [
        init_MetaNCA_task_net_optimized(
            arch,
            task_net_rand_keys[i],
            d_model_layer=d_model_layer,
            d_model_neuron=d_model_neuron,
            d_model_position=d_model_position,
            no_bias=no_bias,
        )
        for i, arch in enumerate(archs)
    ]


def reset_task_nets_sample_pooled(
    archs,
    task_net_params_list,
    task_net_rand_keys,
    d_model_layer,
    d_model_neuron,
    d_model_position,
    zero_out=0.1,
    zero_out_layer=False,
    sample_chance=0.0,
    no_bias=False,
):
    # assume task net params list has two of each arch already; each first one will be replaced and the second one will be kept
    new_task_net_params_list = []
    for i in range(len(archs)):

        task_net_params = task_net_params_list[i * 2]
        task_net_rand_key = task_net_rand_keys[i * 2]
        # task_net_params = task_net_params_list[i]
        # task_net_rand_key = task_net_rand_keys[i]

        sample_old = jax.random.uniform(task_net_rand_key, shape=()) < sample_chance
        if sample_old:
            if zero_out != 0:
                zero_out_rand_key, _ = random.split(task_net_rand_key)
                new_task_net_params = zero_out_weights(
                    task_net_params,
                    zero_out_rand_key,
                    prop_cells_updated=zero_out,
                    zero_out_layer=zero_out_layer,
                    no_bias=no_bias,
                )
            else:
                new_task_net_params = task_net_params
            new_task_net_params_list.append(new_task_net_params)
        else:
            new_init_task_net_params = init_MetaNCA_task_net_optimized(
                archs[i],
                task_net_rand_keys[2 * i + 1],
                d_model_layer=d_model_layer,
                d_model_neuron=d_model_neuron,
                d_model_position=d_model_position,
                no_bias=no_bias,
            )
            new_task_net_params_list.append(
                new_init_task_net_params
            )  # add newly initialized task net

        # sample new ones regardless
        new_init_task_net_params = init_MetaNCA_task_net_optimized(
            archs[i],
            task_net_rand_keys[2 * i + 1],
            d_model_layer=d_model_layer,
            d_model_neuron=d_model_neuron,
            d_model_position=d_model_position,
            no_bias=no_bias,
        )
        new_task_net_params_list.append(new_init_task_net_params)  # add newly initialized task net

    return new_task_net_params_list  # so every even one is possibly kept and every odd one is new


def flatten_params(params):
    """
    Given a params dictionary with keys 'conv_layers' and 'dense_layers', where each layer
    is a dict with keys 'weights' and 'hidden_states', this function returns:
      - flattened_params: a jax.numpy array of shape (total_num_weights, d+1)
         where each row is [weight, hidden_state_vector] (with d elements).
      - weight_ind_to_flat_ind: a jax.numpy array of shape
         (num_layers, max_out_dim, max_in_dim, max_kernel_x, max_kernel_y)
         where each valid weight location stores its flattened index and unused entries are -1.
    """
    layers = []
    layer_types = []

    # First, collect layers from conv and dense parts.
    if "conv_layers" in params:
        for layer in params["conv_layers"]:
            layers.append(layer)
            layer_types.append("conv")
    if "dense_layers" in params:
        for layer in params["dense_layers"]:
            layers.append(layer)
            layer_types.append("dense")

    num_layers = len(layers)

    # Determine maximum dimensions over all layers.
    max_out_dim = 0
    max_in_dim = 0
    max_kernel_x = 0
    max_kernel_y = 0
    for layer, ltype in zip(layers, layer_types):
        weights = layer["weights"]
        if ltype == "conv":
            # Expected shape: (out_dim, in_dim, kx, ky)
            out_dim, in_dim, kx, ky = weights.shape
        else:  # 'dense'
            # Expected shape: (out_dim, in_dim) -> treat as kernel 1x1.
            out_dim, in_dim = weights.shape
            kx, ky = 1, 1
        max_out_dim = max(max_out_dim, out_dim)
        max_in_dim = max(max_in_dim, in_dim)
        max_kernel_x = max(max_kernel_x, kx)
        max_kernel_y = max(max_kernel_y, ky)

    # Determine d (dimension of the hidden state).
    # We assume the hidden_states always have the last dimension = d.
    first_layer = layers[0]
    # For conv layers: shape (out_dim, in_dim, kx, ky, d); for dense: (out_dim, in_dim, d)
    first_layer["hidden_states_w"].shape[-1]

    # Initialize weight_ind_to_flat_ind with -1's.
    weight_ind_to_flat_ind = -jnp.ones(
        (num_layers, max_out_dim, max_in_dim, max_kernel_x, max_kernel_y), dtype=jnp.int32
    )

    flat_params_list = []
    flat_index = 0  # Running counter for the flattened index.

    # Process each layer.
    for layer_idx, (layer, ltype) in enumerate(zip(layers, layer_types)):
        weights = layer["weights"]
        hidden_states = layer["hidden_states_w"]

        if ltype == "conv":
            # weights shape: (out_dim, in_dim, kx, ky)
            out_dim, in_dim, kx, ky = weights.shape
        else:
            # For dense layers, reshape weights to (out_dim, in_dim, 1, 1)
            out_dim, in_dim = weights.shape
            kx, ky = 1, 1
            weights = weights[:, :, None, None]
            # Also reshape hidden_states from (out_dim, in_dim, d) to (out_dim, in_dim, 1, 1, d)
            hidden_states = hidden_states[:, :, None, None, :]

        # Iterate over valid indices in the current layer.
        for i in range(out_dim):
            for j in range(in_dim):
                for a in range(kx):
                    for b in range(ky):
                        # Get the scalar weight and its corresponding hidden state vector.
                        w_val = weights[i, j, a, b]
                        hs_val = hidden_states[i, j, a, b]  # shape (d,)

                        # Combine into a vector of shape (d+1,) with weight first.
                        combined = jnp.concatenate([jnp.array([w_val]), hs_val], axis=0)
                        flat_params_list.append(combined)

                        # Record the flattened index in the mapping.
                        weight_ind_to_flat_ind = weight_ind_to_flat_ind.at[
                            layer_idx, i, j, a, b
                        ].set(flat_index)
                        flat_index += 1

    # Stack the list to form a jax.numpy array.
    flattened_params = jnp.stack(flat_params_list, axis=0)

    return flattened_params, weight_ind_to_flat_ind


# def init_MetaNCA_task_net_convolution(arch, key, d_model_layer=2, d_model_neuron=2, no_bias=False):
def init_MetaNCA_task_net_optimized(
    arch, key, d_model_layer=2, d_model_neuron=2, d_model_position=2, no_bias=False
):
    """
    FOR CONVOLUTION
    """
    (
        conv_layers,
        dense_layer_widths,
    ) = arch  # [(5 filters, 3 channels, 24x24, max_pool_size, max_pool_strides), (24 filters, 5 channels, 3x3, max_pool_size, max_pool_strides)], [20, 10, output_dim]
    len(conv_layers) + len(dense_layer_widths) - 1
    # conv_layer_max_dims = [max(conv_layer) for conv_layer in conv_layers]
    max_conv_layer_dim = 0
    max_height = 0
    max_width = 0
    for conv_layer in conv_layers:
        max_conv_layer_dim = max(
            max_conv_layer_dim, conv_layer[0]
        )  # just max over the filters and channels for now
        max_height = max(max_height, conv_layer[2])
        max_width = max(max_width, conv_layer[3])
    # params = []
    max_layers = len(conv_layers) + len(dense_layer_widths)
    max_neurons = max(max(dense_layer_widths), max_conv_layer_dim)

    layer_encodings = positional_encoding(max_layers, d_model_layer)
    neuron_encodings = positional_encoding(max_neurons, d_model_neuron)
    pos_enc_X_all = positional_encoding(max_width, d_model_position)
    pos_enc_Y_all = positional_encoding(max_height, d_model_position)

    zero_pos_enc = positional_encoding(1, d_model_position)[0]

    # weight_enc_dim = d_model_layer + 2 * d_model_neuron
    conv_keys = random.split(key, len(conv_layers))
    dense_keys = random.split(key, len(dense_layer_widths) - 1)

    def compute_weight_encodings(prev_layer_neuron, next_layer_neuron, layer_encoding):
        # combined_encoding = jnp.concatenate((layer_encoding, neuron_encodings[prev_layer_neuron],
        # neuron_encodings[next_layer_neuron]))
        combined_encoding = jnp.concatenate(
            (
                layer_encoding,
                neuron_encodings[next_layer_neuron],
                neuron_encodings[prev_layer_neuron],
            )
        )
        return combined_encoding

    def compute_bias_encodings(next_layer_neuron, layer_encoding):
        # combined_encoding = jnp.concatenate((layer_encoding, jnp.zeros_like(neuron_encodings[0]), neuron_encodings[next_layer_neuron])) # give all 0s for prev_layer_neurons for biases (no previous layer for biases)
        combined_encoding = jnp.concatenate(
            (
                layer_encoding,
                neuron_encodings[next_layer_neuron],
                jnp.zeros_like(neuron_encodings[0]) - 10,
            )
        )  # give all -10s for prev_layer_neurons for biases (no previous layer for biases)
        return combined_encoding

    def conv_layer_fn(carry, inputs):
        filter_num, channel_num, height, width, max_pool_size_arr, max_pool_strides_arr, k = inputs
        weight_key, bias_key = random.split(k, 2)
        layer_encoding = layer_encodings[carry]
        prev_layer_neurons = jnp.arange(channel_num)
        next_layer_neurons = jnp.arange(filter_num)

        prev_layer_neuron_encodings = neuron_encodings[prev_layer_neurons].reshape(
            1, channel_num, 1, 1, -1
        )
        next_layer_neuron_encodings = neuron_encodings[next_layer_neurons].reshape(
            filter_num, 1, 1, 1, -1
        )

        y_pos_encodings = pos_enc_Y_all[:height].reshape(1, 1, height, 1, -1)
        x_pos_encodings = pos_enc_X_all[:width].reshape(1, 1, 1, width, -1)

        layer_encoding = jnp.broadcast_to(
            layer_encoding, (filter_num, channel_num, height, width, layer_encoding.shape[-1])
        )
        prev_layer_neuron_encodings = jnp.broadcast_to(
            prev_layer_neuron_encodings,
            (filter_num, channel_num, height, width, prev_layer_neuron_encodings.shape[-1]),
        )
        next_layer_neuron_encodings = jnp.broadcast_to(
            next_layer_neuron_encodings,
            (filter_num, channel_num, height, width, next_layer_neuron_encodings.shape[-1]),
        )
        y_pos_encodings = jnp.broadcast_to(
            y_pos_encodings, (filter_num, channel_num, height, width, y_pos_encodings.shape[-1])
        )
        x_pos_encodings = jnp.broadcast_to(
            x_pos_encodings, (filter_num, channel_num, height, width, x_pos_encodings.shape[-1])
        )

        weight_encodings = jnp.concatenate(
            [
                layer_encoding,
                next_layer_neuron_encodings,
                prev_layer_neuron_encodings,
                x_pos_encodings,
                y_pos_encodings,
            ],
            axis=-1,
        )

        hidden_states_w = weight_encodings

        # if carry == 0: # make first layer have weight encodings, otherwise 0
        #    first_layer_weight_encodings = jnp.concatenate([
        #        layer_encoding,
        #        jnp.zeros_like(next_layer_neuron_encodings),
        #        prev_layer_neuron_encodings,
        #        jnp.zeros_like(x_pos_encodings),
        #        jnp.zeros_like(y_pos_encodings)
        #    ], axis=-1)
        #    hidden_states_w = first_layer_weight_encodings

        # else:
        #    hidden_states_w = jnp.zeros_like(weight_encodings)

        # weight_encodings = vmap(vmap(compute_weight_encodings, in_axes=(0, None, None)), in_axes=(None, 0, None))(prev_layer_neurons, next_layer_neurons, layer_encoding)
        if no_bias:
            # hidden_states_w = weight_encodings
            param = {
                "weights": random.normal(weight_key, (filter_num, channel_num, height, width))
                * jnp.sqrt(
                    2 / (channel_num * height * width)
                ),  # to account for number of input units
                # "weights": random.normal(k, (n, m)) * 0.000001, # try tiny weight initialized
                # "hidden_states_w": jnp.tile(weight_encodings[:, :, None, None, :], (1, 1, height, width, 1)),
                # "hidden_states_w_pos_enc": jnp.tile(weight_encodings[:, :, None, None, :], (1, 1, height, width, 1)),
                # "hidden_states_w": weight_encodings,
                "hidden_states_w": hidden_states_w,
                # "hidden_states_w": jnp.ones_like(weight_encodings),
                # "hidden_states_w": random.normal(weight_key, (filter_num, channel_num, height, width, weight_enc_dim)) * jnp.sqrt(2 / weight_enc_dim),
                "hidden_states_w_pos_enc": weight_encodings,
                "max_pool_size": max_pool_size_arr,
                "max_pool_strides": max_pool_strides_arr,
            }
        else:
            # bias_encodings = vmap(compute_bias_encodings, in_axes=(0, None))(next_layer_neurons, layer_encoding)
            bias_layer_encoding = layer_encodings[carry].reshape(1, -1)
            bias_next_neuron_encodings = neuron_encodings[next_layer_neurons]
            bias_prev_neuron_encodings = jnp.full_like(
                bias_next_neuron_encodings, -10.0
            )  # or another appropriate value

            bias_y_pos_encodings = pos_enc_Y_all[0].reshape(1, -1)
            bias_x_pos_encodings = pos_enc_X_all[0].reshape(1, -1)

            bias_encodings = jnp.concatenate(
                [
                    bias_layer_encoding.repeat(filter_num, axis=0),
                    bias_next_neuron_encodings,
                    bias_prev_neuron_encodings,
                    bias_y_pos_encodings.repeat(filter_num, axis=0),
                    bias_x_pos_encodings.repeat(filter_num, axis=0),
                ],
                axis=-1,
            )
            hidden_states_b = jnp.zeros_like(bias_encodings)

            # if carry == 0: # make first layer have weight encodings, otherwise 0
            #    hidden_states_w = weight_encodings
            #    hidden_states_b = bias_encodings
            # else:
            #    hidden_states_w = jnp.zeros_like(weight_encodings)
            #    hidden_states_b = jnp.zeros_like(bias_encodings)
            param = {
                "weights": random.normal(weight_key, (filter_num, channel_num, height, width))
                * jnp.sqrt(
                    2 / (channel_num * height * width)
                ),  # to account for number of input units
                # "weights": random.normal(k, (n, m)) * 0.000001, # try tiny weight initialized
                # "hidden_states_w": jnp.tile(weight_encodings[:, :, None, None, :], (1, 1, height, width, 1)),
                # "hidden_states_w_pos_enc": jnp.tile(weight_encodings[:, :, None, None, :], (1, 1, height, width, 1)),
                # "biases": random.normal(bias_key, (filter_num,)) * jnp.sqrt(2/channel_num), # output_dim
                # "hidden_states_w": weight_encodings,
                "hidden_states_w": hidden_states_w,
                # "hidden_states_w": jnp.ones_like(weight_encodings),
                # "hidden_states_w": random.normal(weight_key, (filter_num, channel_num, height, width, weight_enc_dim)) * jnp.sqrt(2 / weight_enc_dim),
                "hidden_states_w_pos_enc": weight_encodings,
                "biases": jnp.zeros((filter_num,)),  # output_dim
                # "hidden_states_b": bias_encodings,
                "hidden_states_b": hidden_states_b,
                # "hidden_states_b": jnp.ones_like(bias_encodings),
                "hidden_states_b_pos_enc": bias_encodings,
                "max_pool_size": max_pool_size_arr,
                "max_pool_strides": max_pool_strides_arr,
            }
        return carry + 1, param

    def layer_fn(carry, inputs):
        m, n, k = inputs
        weight_key, bias_key = random.split(k, 2)
        layer_encoding = layer_encodings[carry]
        prev_layer_neurons = jnp.arange(m)
        next_layer_neurons = jnp.arange(n)
        # weight_encodings = vmap(vmap(compute_weight_encodings, in_axes=(0, None, None)), in_axes=(None, 0, None))(prev_layer_neurons, next_layer_neurons, layer_encoding)

        prev_layer_neuron_encodings = neuron_encodings[prev_layer_neurons].reshape(1, m, -1)
        next_layer_neuron_encodings = neuron_encodings[next_layer_neurons].reshape(n, 1, -1)

        x_pos_encodings = zero_pos_enc.reshape(1, 1, -1)
        y_pos_encodings = zero_pos_enc.reshape(1, 1, -1)

        # Broadcasting to (n, m, D)
        layer_encoding = jnp.broadcast_to(layer_encoding, (n, m, layer_encoding.shape[-1]))
        prev_layer_neuron_encodings = jnp.broadcast_to(
            prev_layer_neuron_encodings, (n, m, prev_layer_neuron_encodings.shape[-1])
        )
        next_layer_neuron_encodings = jnp.broadcast_to(
            next_layer_neuron_encodings, (n, m, next_layer_neuron_encodings.shape[-1])
        )
        x_pos_encodings = jnp.broadcast_to(x_pos_encodings, (n, m, x_pos_encodings.shape[-1]))
        y_pos_encodings = jnp.broadcast_to(y_pos_encodings, (n, m, y_pos_encodings.shape[-1]))

        weight_encodings = jnp.concatenate(
            [
                layer_encoding,
                next_layer_neuron_encodings,
                prev_layer_neuron_encodings,
                x_pos_encodings,
                y_pos_encodings,
            ],
            axis=-1,
        )

        hidden_states_w = weight_encodings

        if no_bias:
            # hidden_states_w = weight_encodings
            param = {
                "weights": random.normal(weight_key, (n, m)) * jnp.sqrt(2 / m),
                # "weights": random.normal(k, (n, m)) * 0.000001, # try tiny weight initialized
                # "hidden_states_w": weight_encodings,
                "hidden_states_w": hidden_states_w,
                # "hidden_states_w": jnp.ones_like(weight_encodings),
                # "hidden_states_w": random.normal(weight_key, (n, m, weight_enc_dim)) * jnp.sqrt(2 / weight_enc_dim),
                "hidden_states_w_pos_enc": weight_encodings,
            }
        else:
            # bias_encodings = vmap(compute_bias_encodings, in_axes=(0, None))(next_layer_neurons, layer_encoding)
            bias_layer_encoding = layer_encodings[carry].reshape(1, -1)
            bias_next_neuron_encodings = neuron_encodings[next_layer_neurons]
            bias_prev_neuron_encodings = jnp.full_like(
                bias_next_neuron_encodings, -10.0
            )  # or another appropriate value

            bias_x_pos_encodings = zero_pos_enc.reshape(1, -1)
            bias_y_pos_encodings = zero_pos_enc.reshape(1, -1)

            bias_encodings = jnp.concatenate(
                [
                    bias_layer_encoding.repeat(n, axis=0),
                    bias_next_neuron_encodings,
                    bias_prev_neuron_encodings,
                    bias_y_pos_encodings.repeat(n, axis=0),
                    bias_x_pos_encodings.repeat(n, axis=0),
                ],
                axis=-1,
            )
            hidden_states_b = jnp.zeros_like(bias_encodings)

            param = {
                "weights": random.normal(weight_key, (n, m)) * jnp.sqrt(2 / m),
                # "weights": random.normal(k, (n, m)) * 0.000001, # try tiny weight initialized
                # "hidden_states_w": weight_encodings,
                "hidden_states_w": hidden_states_w,
                # "hidden_states_w": jnp.ones_like(weight_encodings),
                # "hidden_states_w": random.normal(weight_key, (n, m, weight_enc_dim)) * jnp.sqrt(2 / weight_enc_dim),
                # "biases": random.normal(bias_key, (n,)) * jnp.sqrt(2/m), # output_dim
                "biases": jnp.zeros(
                    n,
                ),  # output_dim
                # "hidden_states_b": bias_encodings,
                "hidden_states_b": hidden_states_b,
                # "hidden_states_b": jnp.ones_like(bias_encodings),
                "hidden_states_w_pos_enc": weight_encodings,
                "hidden_states_b_pos_enc": bias_encodings,
            }
        return carry + 1, param

    params = {"conv_layers": [], "dense_layers": []}

    for layer_idx, (
        filter_num,
        channel_num,
        height,
        width,
        max_pool_size,
        max_pool_strides,
    ) in enumerate(conv_layers):
        max_pool_size_arr = jnp.zeros((max_pool_size, max_pool_size))
        max_pool_strides_arr = jnp.zeros((max_pool_strides, max_pool_strides))
        param = conv_layer_fn(
            layer_idx,
            (
                filter_num,
                channel_num,
                height,
                width,
                max_pool_size_arr,
                max_pool_strides_arr,
                conv_keys[layer_idx],
            ),
        )[1]
        params["conv_layers"].append(param)

    dense_layer_inputs = list(zip(dense_layer_widths[:-1], dense_layer_widths[1:], dense_keys))
    for layer_idx, (m, n, k) in enumerate(dense_layer_inputs):
        param = layer_fn(layer_idx + len(conv_layers), (m, n, k))[1]
        params["dense_layers"].append(param)

    # flattened_params, weight_ind_to_flat_ind = flatten_params(params)
    # test_flattened_params(params, flattened_params, weight_ind_to_flat_ind)
    # params['flattened_params'] = flattened_params
    # params['weight_ind_to_flat_ind'] = weight_ind_to_flat_ind
    # forward_neighbors, backward_neighbors = compute_neighbor_indices_efficient(params, weight_ind_to_flat_ind)
    # params['flat_forward_neighbors'] = forward_neighbors
    # params['flat_backward_neighbors'] = backward_neighbors

    return params


def test_flattened_params(params, flattened_params, weight_ind_to_flat_ind):
    """
    Test that for every valid index in each layer, the corresponding entry in
    flattened_params matches the concatenation of the weight and hidden state.

    Parameters:
      - params: the original parameters dictionary with keys 'conv_layers' and/or 'dense_layers'.
      - flattened_params: a jax.numpy array of shape (total_num_weights, d+1)
      - weight_ind_to_flat_ind: a jax.numpy array of shape
           (num_layers, max_out_dim, max_in_dim, max_kernel_x, max_kernel_y)
           that maps a layer's local indices to the flattened index.

    Raises an AssertionError if any mismatch is found.
    """
    # Reconstruct the combined list of layers and their types in the same order as flatten_params.
    layers = []
    layer_types = []
    if "conv_layers" in params:
        for layer in params["conv_layers"]:
            layers.append(layer)
            layer_types.append("conv")
    if "dense_layers" in params:
        for layer in params["dense_layers"]:
            layers.append(layer)
            layer_types.append("dense")

    # Loop over each layer.
    for layer_idx, (layer, ltype) in enumerate(zip(layers, layer_types)):
        if ltype == "conv":
            # weights shape: (out_dim, in_dim, kx, ky)
            # hidden_states shape: (out_dim, in_dim, kx, ky, d)
            weights = layer["weights"]
            hidden_states = layer["hidden_states_w"]
            out_dim, in_dim, kx, ky = weights.shape
        else:
            # For dense layers, original weights: (out_dim, in_dim)
            # original hidden_states: (out_dim, in_dim, d)
            # In flatten_params, these were reshaped to (out_dim, in_dim, 1, 1) and (out_dim, in_dim, 1, 1, d)
            weights = layer["weights"]
            hidden_states = layer["hidden_states_w"]
            out_dim, in_dim = weights.shape
            kx, ky = 1, 1

        # Iterate over valid indices in this layer.
        for i in range(out_dim):
            for j in range(in_dim):
                for a in range(kx):
                    for b in range(ky):
                        flat_idx = weight_ind_to_flat_ind[layer_idx, i, j, a, b]
                        # Retrieve the original weight and hidden state.
                        if ltype == "dense":
                            # For dense layers, there's only one valid kernel position (0, 0).
                            weight_val = weights[i, j]
                            hidden_val = hidden_states[i, j]  # shape: (d,)
                        else:
                            weight_val = weights[i, j, a, b]
                            hidden_val = hidden_states[i, j, a, b]  # shape: (d,)
                        # Combine the weight (as a scalar turned into a one-element array) with its hidden state.
                        combined = jnp.concatenate([jnp.array([weight_val]), hidden_val], axis=0)

                        # Assert that the corresponding flattened_params entry is equal to the combined vector.
                        # Using jnp.allclose to allow for any small floating-point differences.
                        # assert jnp.allclose(flattened_params[flat_idx], combined), (
                        assert jnp.all(
                            flattened_params[flat_idx] == combined
                        ), f"Mismatch in layer {layer_idx} at index (i={i}, j={j}, a={a}, b={b})."
    print("All tests passed!")


def calculate_max_num_neighbors(params, weight_ind_to_flat_ind):
    """
    Loop through all layers and their valid weight indices to compute the
    maximum possible number of neighbors (forward + backward) any weight can have.
    Uses the connection rules:
      - Intra-layer backward neighbors: all weights with same input index (j) except itself.
      - Intra-layer forward neighbors: all weights with same output index (i) except itself.
      - Cross-layer backward: if there's a previous layer, all weights in that layer whose output index equals current weight's input index.
      - Cross-layer forward: if there's a next layer, all weights in that layer whose input index equals current weight's output index.
    """
    max_neighbors = 0
    layers = []
    layer_types = []
    if "conv_layers" in params:
        for layer in params["conv_layers"]:
            layers.append(layer)
            layer_types.append("conv")
    if "dense_layers" in params:
        for layer in params["dense_layers"]:
            layers.append(layer)
            layer_types.append("dense")
    num_layers = len(layers)

    for L in range(num_layers):
        layer = layers[L]
        ltype = layer_types[L]
        if ltype == "conv":
            out_dim, in_dim, kx, ky = layer["weights"].shape
        else:
            out_dim, in_dim = layer["weights"].shape
            kx, ky = 1, 1

        # Intra-layer neighbors:
        # For backward: all weights with same input index j (across all output indices and kernel positions)
        intra_backward = out_dim * (kx * ky) - 1  # exclude self
        # For forward: all weights with same output index i
        intra_forward = in_dim * (kx * ky) - 1

        # Cross-layer neighbors:
        cross_backward = 0
        if L > 0:
            prev_layer = layers[L - 1]
            prev_type = layer_types[L - 1]
            if prev_type == "conv":
                prev_out_dim, prev_in_dim, pkx, pky = prev_layer["weights"].shape
            else:
                prev_out_dim, prev_in_dim = prev_layer["weights"].shape
                pkx, pky = 1, 1
            # Maximum: assume the current weight's input index is valid (i.e. < prev_out_dim)
            cross_backward = prev_in_dim * (pkx * pky)

        cross_forward = 0
        if L < num_layers - 1:
            next_layer = layers[L + 1]
            next_type = layer_types[L + 1]
            if next_type == "conv":
                next_out_dim, next_in_dim, nkx, nky = next_layer["weights"].shape
            else:
                next_out_dim, next_in_dim = next_layer["weights"].shape
                nkx, nky = 1, 1
            # Maximum: assume the current weight's output index is valid for next layer
            cross_forward = next_out_dim * (nkx * nky)

        total_neighbors = intra_backward + intra_forward + cross_backward + cross_forward
        max_neighbors = max(max_neighbors, total_neighbors)
    return max_neighbors


def get_intra_layer_neighbors(flat_indices, grid, coords, static_max_neighbors):
    """
    Vectorized neighbor extraction for a single weight in a layer.
    Returns fixed-size arrays for forward and backward neighbors
    (size given by static_max_neighbors, with -1 padding).
    """
    i, j, a, b = coords

    # Intra-layer backward: same input index j, excluding self.
    mask_backward = (grid[:, 1] == j) & ~(
        (grid[:, 0] == i) & (grid[:, 1] == j) & (grid[:, 2] == a) & (grid[:, 3] == b)
    )
    bw_indices = jnp.nonzero(mask_backward, size=static_max_neighbors, fill_value=-1)[0]
    backward_neighbors = jnp.where(bw_indices >= 0, flat_indices[bw_indices], -1)

    # Intra-layer forward: same output index i, excluding self.
    mask_forward = (grid[:, 0] == i) & ~(
        (grid[:, 0] == i) & (grid[:, 1] == j) & (grid[:, 2] == a) & (grid[:, 3] == b)
    )
    fw_indices = jnp.nonzero(mask_forward, size=static_max_neighbors, fill_value=-1)[0]
    forward_neighbors = jnp.where(fw_indices >= 0, flat_indices[fw_indices], -1)

    return forward_neighbors, backward_neighbors


def compute_neighbors_vectorized_intralayer(layer, mapping, ltype, static_max_neighbors):
    """
    Computes the intra-layer forward and backward neighbor indices using vectorized operations.
    Returns dictionaries mapping flattened indices to their fixed-size neighbor arrays.
    """
    if ltype == "conv":
        out_dim, in_dim, kx, ky = layer["weights"].shape
    else:
        out_dim, in_dim = layer["weights"].shape
        kx, ky = 1, 1

    # Create coordinate grid for valid weights.
    i_coords, j_coords, a_coords, b_coords = jnp.meshgrid(
        jnp.arange(out_dim), jnp.arange(in_dim), jnp.arange(kx), jnp.arange(ky), indexing="ij"
    )
    grid = jnp.stack(
        [i_coords, j_coords, a_coords, b_coords], axis=-1
    )  # shape: (out_dim, in_dim, kx, ky, 4)
    grid_flat = grid.reshape(-1, 4)

    # Get the flattened indices for valid weights.
    flat_indices = mapping[:out_dim, :in_dim, :kx, :ky].reshape(-1)

    # Use vmap to compute neighbor indices for each weight.
    get_neighbors_vmap = jax.vmap(
        lambda coords: get_intra_layer_neighbors(
            flat_indices, grid_flat, coords, static_max_neighbors
        )
    )
    forward_neigh_array, backward_neigh_array = get_neighbors_vmap(grid_flat)

    # Build dictionaries keyed by flattened index.
    neighbor_forward = {}
    neighbor_backward = {}
    for idx, f_neigh, b_neigh in zip(flat_indices, forward_neigh_array, backward_neigh_array):
        neighbor_forward[int(idx)] = list(f_neigh)
        neighbor_backward[int(idx)] = list(b_neigh)
    return neighbor_forward, neighbor_backward


def compute_neighbor_indices_efficient(params, weight_ind_to_flat_ind):
    """
    Compute both intra- and cross-layer neighbor indices.
    Uses precomputed static max neighbors (via a loop over layers) for the vmap operations.
    Returns:
      - forward_neighbors: (total_num_weights, static_max_neighbors_total)
      - backward_neighbors: (total_num_weights, static_max_neighbors_total)
    """
    # Precompute static maximum number of neighbors over all layers.
    print("before calculate_max_num_neighbors")
    static_max_neighbors = calculate_max_num_neighbors(params, weight_ind_to_flat_ind)
    print("after calculate_max_num_neighbors")

    layers = []
    layer_types = []
    if "conv_layers" in params:
        for layer in params["conv_layers"]:
            layers.append(layer)
            layer_types.append("conv")
    if "dense_layers" in params:
        for layer in params["dense_layers"]:
            layers.append(layer)
            layer_types.append("dense")
    num_layers = len(layers)

    neighbor_forward_dict = {}
    neighbor_backward_dict = {}

    # Compute intra-layer neighbors vectorized.
    for L, (layer, ltype) in enumerate(zip(layers, layer_types)):
        mapping = weight_ind_to_flat_ind[L]
        print("before compute_neighbors_vectorized_intralayer")
        nf, nb = compute_neighbors_vectorized_intralayer(
            layer, mapping, ltype, static_max_neighbors
        )
        print("after compute_neighbors_vectorized_intralayer")
        for key in nf:
            neighbor_forward_dict[key] = nf[key]
            neighbor_backward_dict[key] = nb[key]

        # Cross-layer neighbors: add backward neighbors from previous layer.
        if L > 0:
            prev_layer = layers[L - 1]
            prev_type = layer_types[L - 1]
            prev_mapping = weight_ind_to_flat_ind[L - 1]
            if prev_type == "conv":
                prev_out_dim, prev_in_dim, pkx, pky = prev_layer["weights"].shape
            else:
                prev_out_dim, prev_in_dim = prev_layer["weights"].shape
                pkx, pky = 1, 1
            if ltype == "conv":
                out_dim, in_dim, kx, ky = layer["weights"].shape
            else:
                out_dim, in_dim = layer["weights"].shape
                kx, ky = 1, 1
            for i in range(out_dim):
                for j in range(in_dim):
                    for a in range(kx):
                        for b in range(ky):
                            cur_flat = int(mapping[i, j, a, b])
                            if j < prev_out_dim:
                                for j_prev in range(prev_in_dim):
                                    for a_prev in range(pkx):
                                        for b_prev in range(pky):
                                            idx_val = int(prev_mapping[j, j_prev, a_prev, b_prev])
                                            if idx_val >= 0:
                                                neighbor_backward_dict[cur_flat].append(idx_val)
        # Cross-layer forward neighbors from next layer.
        if L < num_layers - 1:
            next_layer = layers[L + 1]
            next_type = layer_types[L + 1]
            next_mapping = weight_ind_to_flat_ind[L + 1]
            if next_type == "conv":
                next_out_dim, next_in_dim, nkx, nky = next_layer["weights"].shape
            else:
                next_out_dim, next_in_dim = next_layer["weights"].shape
                nkx, nky = 1, 1
            if ltype == "conv":
                out_dim, in_dim, kx, ky = layer["weights"].shape
            else:
                out_dim, in_dim = layer["weights"].shape
                kx, ky = 1, 1
            for i in range(out_dim):
                for j in range(in_dim):
                    for a in range(kx):
                        for b in range(ky):
                            cur_flat = int(mapping[i, j, a, b])
                            if i < next_in_dim:
                                for i_next in range(next_out_dim):
                                    for a_next in range(nkx):
                                        for b_next in range(nky):
                                            idx_val = int(next_mapping[i_next, i, a_next, b_next])
                                            if idx_val >= 0:
                                                neighbor_forward_dict[cur_flat].append(idx_val)

    total_num_weights = int(jnp.max(weight_ind_to_flat_ind)) + 1
    # Determine a static total neighbor size (maximum length across all weights).
    static_max_neighbors_total = max(
        max(len(neighbor_forward_dict.get(idx, [])), len(neighbor_backward_dict.get(idx, [])))
        for idx in range(total_num_weights)
    )

    forward_neighbors = -jnp.ones((total_num_weights, static_max_neighbors_total), dtype=jnp.int32)
    backward_neighbors = -jnp.ones((total_num_weights, static_max_neighbors_total), dtype=jnp.int32)

    for idx in range(total_num_weights):
        f_list = neighbor_forward_dict.get(idx, [])
        b_list = neighbor_backward_dict.get(idx, [])
        f_list_padded = f_list + [-1] * (static_max_neighbors_total - len(f_list))
        b_list_padded = b_list + [-1] * (static_max_neighbors_total - len(b_list))
        forward_neighbors = forward_neighbors.at[idx].set(jnp.array(f_list_padded, dtype=jnp.int32))
        backward_neighbors = backward_neighbors.at[idx].set(
            jnp.array(b_list_padded, dtype=jnp.int32)
        )

    return forward_neighbors, backward_neighbors


def init_MetaNCA_task_testing(layer_widths, key, d_model_layer=2, d_model_neuron=2):
    params = []
    max_layers = len(layer_widths)
    max_neurons = max(layer_widths)
    layer_encodings = positional_encoding(max_layers, d_model_layer)
    neuron_encodings = positional_encoding(max_neurons, d_model_neuron)
    weight_enc_dim = d_model_layer + 2 * d_model_neuron
    keys = random.split(key, len(layer_widths) - 1)

    def compute_weight_encodings(prev_layer_neuron, next_layer_neuron, layer_encoding):
        combined_encoding = jnp.concatenate(
            (
                layer_encoding,
                neuron_encodings[prev_layer_neuron],
                neuron_encodings[next_layer_neuron],
            )
        )
        return combined_encoding

    def layer_fn(carry, inputs):
        m, n, k = inputs
        layer_encoding = layer_encodings[carry]
        prev_layer_neurons = jnp.arange(m)
        next_layer_neurons = jnp.arange(n)
        weight_encodings = vmap(
            vmap(compute_weight_encodings, in_axes=(0, None, None)), in_axes=(None, 0, None)
        )(prev_layer_neurons, next_layer_neurons, layer_encoding)
        params = {
            # "weights": random.normal(k, (n, m)) * jnp.sqrt(2 / m),
            "hidden_states": weight_encodings
        }
        return carry + 1, params

    # layer_inputs = jnp.array(list(zip(layer_widths[:-1], layer_widths[1:])))
    layer_inputs = list(zip(layer_widths[:-1], layer_widths[1:], keys))
    # _, params = lax.scan(layer_fn, (0, None), layer_inputs)
    for layer_idx, (m, n, k) in enumerate(layer_inputs):
        # params.append(layer_fn(layer_idx, (m, n, k))[1])
        params.append({"hidden_states": jnp.zeros((n, m, weight_enc_dim))})

        if layer_idx == 0:
            params[layer_idx]["weights"] = jnp.concatenate(
                [jnp.ones((1, m)), jnp.zeros((n - 1, m))]
            )
        else:
            all_internal_neighbors = jnp.zeros((n, m))
            # all_internal_neighbors = all_internal_neighbors.at[1:, 0].add(jnp.ones((n-1)))
            all_internal_neighbors = all_internal_neighbors.at[:, 0].add(jnp.ones((n)))
            all_internal_neighbors = all_internal_neighbors.at[0, 1:].add(jnp.ones((m - 1)))
            params[layer_idx]["weights"] = all_internal_neighbors
    return params


def init_MetaNCA_task_net(layer_widths, key, d_model_layer=2, d_model_neuron=2):
    """
    input: layer_widths: list of dimensions of feedforward network neurons. Smallest possible network is [4,3] for iris datset.
    key: random key for jax pseudorandom number generation
    d_model_layer: number of dimensions of hidden state to initially have layer positional encoding
    d_model_neuron: number of dimensions of hidden state to initially have neuron positional encoding, for each weight index i and j
    Total hidden_dim will be given by those two dimensions: d_model_layer + 2*d_model_neuron
    """
    params = []
    max_layers = len(layer_widths)
    max_neurons = max(layer_widths)
    layer_encodings = positional_encoding(max_layers, d_model_layer)
    neuron_encodings = positional_encoding(max_neurons, d_model_neuron)
    weight_enc_dim = d_model_layer + 2 * d_model_neuron

    keys = random.split(key, len(layer_widths) - 1)
    for i, (m, n, k) in enumerate(zip(layer_widths[:-1], layer_widths[1:], keys)):
        layer_encoding = layer_encodings[i]
        weight_encodings = jnp.zeros(shape=(n, m, weight_enc_dim))
        for prev_layer_neuron in range(m):
            for next_layer_neuron in range(n):
                combined_encoding = jnp.concatenate(
                    (
                        layer_encoding,
                        neuron_encodings[prev_layer_neuron],
                        neuron_encodings[next_layer_neuron],
                    )
                )
                weight_encodings = weight_encodings.at[next_layer_neuron, prev_layer_neuron].set(
                    combined_encoding
                )
        params.append(
            {
                "weights": random.normal(k, (n, m))
                * jnp.sqrt(2 / m),  # output_dim, input_dim for convention
                # "weights": random.normal(weight_key, (n, m)) * jnp.sqrt(2/m),
                # "biases": random.normal(bias_key, (n,)) * jnp.sqrt(2/m), # output_dim
                "hidden_states": weight_encodings,  # output_dim, input_dim
            }
        )
    return params


def init_local_rule_net(
    layer_widths: list[int],
    hidden_dim: int,
    key: jax.random.PRNGKey,
    n_spatial_dims: int = 2,
    weight_transformer_dropout: float = 0.0,
) -> tuple[LocalRuleNet, chex.ArrayTree]:

    local_rule_net = LocalRuleNet(
        hidden_dim=hidden_dim,
        local_rule_layer_widths=tuple(layer_widths),
        weight_transformer_dropout=weight_transformer_dropout,
        n_spatial_dims=n_spatial_dims,
        bias_linear_attn=False,
        bias_local_rule=True,
    )

    state_dim = hidden_dim + 1
    dummy_focus = jnp.zeros((1, 1, state_dim))
    dummy_neighbors = jnp.zeros((1, 1, state_dim))
    dummy_pos_enc = jnp.zeros((1, 1, hidden_dim))
    dummy_mask = jnp.ones((1, 1), dtype=bool)

    params = local_rule_net.init(
        key,
        dummy_focus,
        dummy_focus,
        dummy_neighbors,
        dummy_neighbors,
        dummy_pos_enc,
        dummy_pos_enc,
        dummy_pos_enc,
        dummy_pos_enc,
        dummy_mask,
        dummy_mask,
        1,
        deterministic=True,
    )

    return local_rule_net, params


'''

def init_local_rule_net(layer_widths, hidden_dim, key):
    """
    Initializes parameters for the Local Rule Network with GRU.

    Args:
        layer_widths (list of int): Sizes of each layer in the LRN.
        hidden_dim (int): Dimension of the GRU's hidden state.
        key (jax.random.PRNGKey): Random key for initialization.

    Returns:
        params (dict): Dictionary containing LRN and GRU parameters.
    """
    params = {}

    """
    # Initialize LRN layers
    lrn_params = []
    keys = random.split(key, len(layer_widths) - 1)
    for i, (m, n, k) in enumerate(zip(layer_widths[:-1], layer_widths[1:], keys)):
        weight_key, bias_key = random.split(k, 2)
        if i == len(layer_widths) - 2:
            # Final layer: initialize weights to zeros
            lrn_params.append({
                "weights": jnp.zeros((n, m)),
                "biases": jnp.zeros((n,))
            })
        else:
            # Hidden layers: He initialization for ReLU
            lrn_params.append({
                "weights": random.normal(weight_key, (n, m)) * jnp.sqrt(2 / m),
                "biases": jnp.zeros(n)  # Initialize biases to zero for ReLU
            })
    params['lrn'] = lrn_params
    """

    # Initialize GRU parameters
    rng_z, rng_r, rng_h = random.split(key, 3)
    input_dim = 3*(hidden_dim + 1) # forward, current and back

    gru_params = {
        "W_z": random.normal(rng_z, (hidden_dim, hidden_dim + input_dim)) * jnp.sqrt(2 / (hidden_dim + input_dim)),
        "b_z": jnp.zeros(hidden_dim),
        "W_r": random.normal(rng_r, (hidden_dim, hidden_dim + input_dim)) * jnp.sqrt(2 / (hidden_dim + input_dim)),
        "b_r": jnp.zeros(hidden_dim),
        "W_h": random.normal(rng_h, (hidden_dim, hidden_dim + input_dim)) * jnp.sqrt(2 / (hidden_dim + input_dim)),
        "b_h": jnp.zeros(hidden_dim),
        "W_out": jnp.zeros((1, hidden_dim))  # Initialize to zeros to prevent initial changes
    }
    params['gru'] = gru_params

    return params
'''


def init_simple_local_rule_net(layer_widths):
    params = []
    for i, (m, n) in enumerate(zip(layer_widths[:-1], layer_widths[1:])):
        params.append({"weights": jnp.ones((n, m)), "biases": jnp.zeros((n,))})
    return params


def zero_out_weights(
    task_net_params, rand_key, prop_cells_updated=0, zero_out_layer=False, no_bias=False
):
    new_task_net_params = []
    zero_layer_idx = -1
    if zero_out_layer:
        zero_layer_rand_key, rand_key = random.split(rand_key)
        zero_layer_idx = random.choice(rand_key, jnp.arange(len(task_net_params)))
    for param_idx, param in enumerate(task_net_params):
        weights = param["weights"]
        hidden_states_w = param["hidden_states_w"]
        hidden_states_w_pos_enc = param["hidden_states_w_pos_enc"]
        if not no_bias:
            biases = param["biases"]
            hidden_states_b = param["hidden_states_b"]
            hidden_states_b_pos_enc = param["hidden_states_b_pos_enc"]
        if param_idx == zero_layer_idx:
            param_dict = {
                "weights": jnp.zeros_like(weights),
                "hidden_states_w": hidden_states_w,
                "hidden_states_w_pos_enc": hidden_states_w_pos_enc,
            }
            if not no_bias:
                param_dict["biases"] = jnp.zeros_like(biases)
                param_dict["hidden_states_b"] = hidden_states_b
                param_dict["hidden_states_b_pos_enc"] = hidden_states_b_pos_enc
            new_task_net_params.append(param_dict)
        else:
            rand_key, prop_cell_key = jax.random.split(rand_key, 2)
            # Using jax.numpy to construct a new parameter array
            total_num_cells = weights.shape[0] * weights.shape[1]
            flat_indices = random.choice(
                prop_cell_key,
                total_num_cells,
                shape=(int(total_num_cells * prop_cells_updated),),
                replace=False,
            )
            update_inds_i = flat_indices // weights.shape[1]
            update_inds_j = flat_indices % weights.shape[1]

            new_weights = weights.at[update_inds_i, update_inds_j].set(0)

            param_dict = {
                "weights": new_weights,
                "hidden_states_w": hidden_states_w,
                "hidden_states_w_pos_enc": hidden_states_w_pos_enc,
            }
            if not no_bias:

                update_inds = random.choice(
                    prop_cell_key,
                    biases.shape[0],
                    shape=(int(biases.shape[0] * prop_cells_updated),),
                    replace=False,
                )
                new_biases = biases.at[update_inds].set(0)
                param_dict["biases"] = jnp.array(new_biases)
                param_dict["hidden_states_b"] = hidden_states_b
                param_dict["hidden_states_b_pos_enc"] = hidden_states_b_pos_enc
            new_task_net_params.append(param_dict)
    return new_task_net_params
