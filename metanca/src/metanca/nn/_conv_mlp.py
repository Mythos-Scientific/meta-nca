import flax.linen as nn
import jax

from metanca.typing import ConvLayerSpec, MLPLayerSpec, Padding

from ._conv_block import ConvBlock
from ._multi_layer_perceptron import MultiLayerPerceptron


class ConvMLP(nn.Module):

    conv_layer_specs: list[ConvLayerSpec]
    mlp_layer_specs: list[MLPLayerSpec]

    mlp_use_bias: bool = True
    conv_use_bias: bool = True
    padding: Padding = "VALID"
    strides: int = 1

    activation: str = "leaky_relu"

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        conv_block = ConvBlock(
            self.conv_layer_specs,
            use_bias=self.conv_use_bias,
            padding=self.padding,
            strides=self.strides,
            activation=self.activation,
        )
        mlp = MultiLayerPerceptron(
            self.mlp_layer_specs, use_bias=self.mlp_use_bias, activation=self.activation
        )

        cnn_activations = conv_block(x)
        batch_size = cnn_activations.shape[0]

        flattened = cnn_activations.reshape(batch_size, -1)
        return mlp(flattened)
