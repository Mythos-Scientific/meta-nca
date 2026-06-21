import warnings
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from metanca.functional import get_activation


class ResidualBlock(nn.Module):
    out_channels: int
    kernel_size: int
    strides: int = 1
    use_bn: bool = False
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32
    use_bias: bool = False
    act: Optional[str] = "relu"

    @nn.compact
    def __call__(self, x: jax.Array, *, training: bool) -> jax.Array:
        if self.use_bn:
            warnings.warn(
                "BatchNorm is implemented, but not well tested in parameter graph creation!"
                "This may cause unexpected behaviors.",
                stacklevel=2,
            )

        activation = get_activation(self.act)

        if self.use_bn:

            def layer(x: jax.Array, kernel_size: int, stride: int) -> jax.Array:
                x = nn.Conv(
                    features=self.out_channels,
                    kernel_size=(kernel_size, kernel_size),
                    strides=(stride, stride),
                    padding="SAME",
                    use_bias=self.use_bias,
                    dtype=self.dtype,
                )(x)
                x = nn.BatchNorm(
                    use_running_average=not training,
                    momentum=self.bn_momentum,
                    epsilon=self.bn_epsilon,
                    dtype=self.dtype,
                )(x)
                return x

        else:

            def layer(x: jax.Array, kernel_size: int, stride: int) -> jax.Array:
                x = nn.Conv(
                    features=self.out_channels,
                    kernel_size=(kernel_size, kernel_size),
                    strides=(stride, stride),
                    padding="SAME",
                    use_bias=self.use_bias,
                    dtype=self.dtype,
                )(x)
                return x

        residual = x
        x = activation(layer(x, self.kernel_size, self.strides))
        x = layer(x, self.kernel_size, 1)

        in_channels = residual.shape[-1]
        if self.strides != 1 or in_channels != self.out_channels:
            residual = layer(residual, 1, self.strides)

        x = activation(x + residual)
        return x
