import flax.linen as nn
import jax

from metanca.functional import get_activation
from metanca.typing import ConvLayerSpec, Padding


class ConvBlock(nn.Module):
    layer_specs: list[ConvLayerSpec]
    use_bias: bool = True
    padding: Padding = "VALID"
    strides: int = 1
    activation: str = "leaky_relu"

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        conv_layers = [
            nn.Conv(
                features=features,
                kernel_size=kernel_size,
                strides=self.strides,
                padding=self.padding,
                use_bias=self.use_bias,
            )
            for features, kernel_size, *_ in self.layer_specs
        ]

        activation = x
        act_fn = get_activation(self.activation)
        for layer, (*_, pool_size, pool_strides) in zip(conv_layers, self.layer_specs):
            activation = layer(activation)
            activation = act_fn(activation)
            activation = nn.max_pool(
                activation, window_shape=pool_size, strides=pool_strides, padding="VALID"
            )

        return activation
