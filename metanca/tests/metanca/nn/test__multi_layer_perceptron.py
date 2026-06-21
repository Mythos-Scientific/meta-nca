import jax
import jax.numpy as jnp
import pytest

from metanca.functional import flatten_params
from metanca.nn import MultiLayerPerceptron


@pytest.mark.parametrize("layer_spec", [[10, 20, 100], [100, 20, 10]])
def test_multi_layer_perceptron_instantiate(layer_spec: list[int]):

    in_features = 4
    x = jnp.ones((1, in_features))
    key = jax.random.key(0)
    mlp = MultiLayerPerceptron(layer_specs=layer_spec, use_bias=False)
    params = mlp.init(key, x)
    flattened_params = flatten_params(params)

    input_shapes = [in_features] + layer_spec[:-1]
    expected_shapes = zip(input_shapes, layer_spec)
    for (name, parameter), expected_shape in zip(flattened_params, expected_shapes):
        assert parameter.shape == expected_shape
