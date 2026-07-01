import jax
import pytest

import metanca
from metanca_training.scaling.arch_grid import build_mlp
from metanca_training.scaling.hidden_state import grid_hidden_state_initializer

IN = (784,)
PE = dict(d_neuron=10, d_layer=10, d_spatial=10)


def _build_with(widths, shared):
    return metanca.TaskNet.build(
        model=build_mlp(widths, 10), input_shape=IN, key=jax.random.key(1),
        n_spatial_dims=0, shared_initializer=shared, **PE,
    )


def test_grid_initializer_hidden_dim():
    _, hidden_dim = grid_hidden_state_initializer(784, **PE)
    assert hidden_dim == 30  # 2*d_neuron + d_layer (no spatial)


def test_grid_initializer_covers_extreme_archs():
    shared = grid_hidden_state_initializer(784, **PE)
    for widths in [(512, 512, 512, 512, 512), (32, 32, 32, 32, 32), (512, 32), (32, 32)]:
        _build_with(widths, shared)  # must not raise (in-bounds for all depths/widths)


def test_too_shallow_initializer_raises_on_deep_arch():
    # asserts enabled (pytest default); a depth-2 initializer has n_layers=3
    from metanca.hidden_state import hidden_state_initializer
    import jax.numpy as jnp
    shallow = hidden_state_initializer(jnp.array([784, 512]), 3, **PE)
    with pytest.raises(AssertionError):
        _build_with((512, 512, 512, 512, 512), shallow)
