import jax
import jax.numpy as jnp


def local_rule_net_forward(
    local_rule_net_params_list: list, x: jax.Array
) -> tuple[jax.Array, jax.Array]:
    activations = x
    for layer in local_rule_net_params_list[
        1:-1
    ]:  # skip attention layer here, attention handled in compute_aggregated_signals_function
        outputs = jnp.dot(activations, layer["weights"].T) + layer["biases"]
        # activations = relu(outputs)
        activations = jax.nn.elu(outputs)

    final_layer = local_rule_net_params_list[-1]
    # CHANGE THIS BACK
    y_pred = jax.nn.tanh(jnp.dot(activations, final_layer["weights"].T) + final_layer["biases"])
    # final_act = jnp.dot(activations, final_layer["weights"].T) + final_layer["biases"]
    # final_act = nn.tanh(jnp.dot(activations, final_layer["weights"].T) + final_layer["biases"])
    # y_pred = layer_norm(final_act, final_layer['gamma'], final_layer['beta'])
    delta_theta = y_pred[:, 0]
    delta_hidden = y_pred[:, 1:]

    return delta_theta, delta_hidden
