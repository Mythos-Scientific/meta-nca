import warnings
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from metanca.functional import get_activation

from ._residual_block import ResidualBlock


class ResNet(nn.Module):
    num_classes: int
    stage_blocks: tuple[int, int, int]
    base_channels: int = 16
    use_bn: bool = False
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    dtype: Any = jnp.float32
    act: str = "relu"

    @nn.compact
    def __call__(self, x: jax.Array, *, training: bool = False) -> jax.Array:
        if self.use_bn:
            warnings.warn(
                "BatchNorm is implemented, but not well tested in parameter graph creation! "
                "This may cause unexpected behaviors.",
                stacklevel=2,
            )
        x = nn.Conv(
            features=self.base_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            dtype=self.dtype,
            padding="SAME",
        )(x)
        if self.use_bn:
            x = nn.BatchNorm(
                use_running_average=not training,
                momentum=self.bn_momentum,
                epsilon=self.bn_epsilon,
                dtype=self.dtype,
            )(x)

        x = get_activation(self.act)(x)

        channels = self.base_channels
        for stage_idx, n_blocks in enumerate(self.stage_blocks):
            for block_idx in range(n_blocks):
                stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                x = ResidualBlock(
                    out_channels=channels,
                    kernel_size=3,
                    use_bn=self.use_bn,
                    bn_momentum=self.bn_momentum,
                    bn_epsilon=self.bn_epsilon,
                    dtype=self.dtype,
                    strides=stride,
                    act=self.act,
                )(x, training=training)
            channels *= 2

        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(features=self.num_classes, dtype=self.dtype)(x)
        return x
