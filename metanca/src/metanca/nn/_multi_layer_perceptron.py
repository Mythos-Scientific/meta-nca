from typing import Callable

import flax.linen as nn
import jax

from metanca.functional import get_activation
from metanca.typing import MLPLayerSpec


class MultiLayerPerceptron(nn.Module):

    layer_specs: list[MLPLayerSpec]
    activation: str = "leaky_relu"
    final_layer_activation: str = "identity"
    final_layer_kernel_init: Callable[..., jax.Array] | None = None
    final_layer_bias_init: Callable[..., jax.Array] | None = None
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        dense_layers = []
        for i, dim in enumerate(self.layer_specs):
            is_last = i == len(self.layer_specs) - 1
            if is_last and self.final_layer_kernel_init is not None:
                dense_layers.append(
                    nn.Dense(
                        dim,
                        use_bias=self.use_bias,
                        kernel_init=self.final_layer_kernel_init,
                        bias_init=self.final_layer_bias_init or nn.initializers.zeros,
                    )
                )
            else:
                dense_layers.append(nn.Dense(dim, use_bias=self.use_bias))
        activations = ([get_activation(self.activation)] * (len(dense_layers) - 1)) + [
            get_activation(self.final_layer_activation)
        ]

        for layer, activation in zip(dense_layers, activations):
            x = activation(layer(x))

        return x
