import jax

import metanca
from metanca_training.scaling.arch_grid import build_mlp
from metanca_training.scaling.hidden_state import grid_hidden_state_initializer


def test_build_many_uses_shared_initializer():
    shared = grid_hidden_state_initializer(784, d_neuron=10, d_layer=10, d_spatial=10)
    tns = metanca.TaskNet.build_many(
        models=[build_mlp((32,), 10), build_mlp((64, 32), 10)],
        input_shapes=[(784,), (784,)], key=jax.random.key(0),
        d_neuron=10, d_layer=10, d_spatial=10, shared_initializer=shared,
    )
    assert all(tn.hidden_dim == shared[1] for tn in tns)
    assert all(tn.hidden_state_initializer is not None for tn in tns)
