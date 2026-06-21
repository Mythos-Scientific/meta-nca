"""Derived config helpers for Hydra-driven training."""


def compute_local_rule_arch(
    hidden_state_dim: int,
    hidden_layers: list[int],
) -> list[int]:
    """Compute full local rule architecture with input/output dims.

    Input dim: 3 + 3 * hidden_state_dim (from perception)
    Output dim: 1 + hidden_state_dim (delta_weight + delta_hidden)
    """
    input_dim = 3 + 3 * hidden_state_dim
    output_dim = 1 + hidden_state_dim
    return [input_dim] + hidden_layers + [output_dim]


def compute_hidden_state_dim(
    input_shape: tuple[int, ...],
    d_layer: int,
    d_neuron: int,
    d_spatial: int,
) -> int:
    """Compute hidden state dimension from positional encoding dims."""
    *spatial_dims, _ = input_shape
    return (len(spatial_dims) * d_spatial) + 2 * d_neuron + d_layer
