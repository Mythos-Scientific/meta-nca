"""Explicit grid-max hidden-state initializer for the scaling study.

Provisions the positional-encoding tables to the grid-wide maxima (neuron table
max(input_dim, MAX_HIDDEN_WIDTH); layer table MAX_HIDDEN_LAYERS + 1 for the output layer),
identically for training and evaluation. See spec "Hidden-state initialization consistency".
"""

import jax.numpy as jnp

from metanca.hidden_state import hidden_state_initializer

from .arch_grid import MAX_HIDDEN_LAYERS, MAX_HIDDEN_WIDTH


def grid_hidden_state_initializer(
    input_dim: int, d_neuron: int, d_layer: int, d_spatial: int
):
    """Return (initializer_fn, hidden_dim) provisioned to the grid-wide maxima."""
    max_in = max(input_dim, MAX_HIDDEN_WIDTH)   # 784 for flat Fashion-MNIST
    max_out = MAX_HIDDEN_WIDTH                   # 512 (>= n_classes)
    n_layers = MAX_HIDDEN_LAYERS + 1             # + output layer => 6
    return hidden_state_initializer(
        jnp.array([max_in, max_out]),
        n_layers,
        d_neuron=d_neuron, d_layer=d_layer, d_spatial=d_spatial,
    )
