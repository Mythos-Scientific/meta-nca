import jax
import jax.numpy as jnp
import pytest

from metanca.nn.resnet._residual_block import ResidualBlock


@pytest.mark.parametrize(
    ("in_channels", "out_channels", "strides", "use_bn", "expected_shape"),
    [
        (16, 16, 1, False, (2, 32, 32, 16)),
        (16, 32, 2, False, (2, 16, 16, 32)),
        (16, 16, 1, True, (2, 32, 32, 16)),
        (16, 32, 2, True, (2, 16, 16, 32)),
    ],
)
def test_residual_block_output_shape(
    in_channels: int,
    out_channels: int,
    strides: int,
    use_bn: bool,
    expected_shape: tuple[int, int, int, int],
):
    x = jnp.ones((2, 32, 32, in_channels))
    model = ResidualBlock(out_channels=out_channels, kernel_size=3, strides=strides, use_bn=use_bn)
    variables = model.init(jax.random.key(0), x, training=True)
    y = model.apply(variables, x, training=False)
    assert y.shape == expected_shape


def test_residual_block_with_none_activation_uses_identity():
    x = jnp.ones((2, 16, 16, 8))
    model = ResidualBlock(out_channels=8, kernel_size=3, strides=1, use_bn=False, act=None)
    variables = model.init(jax.random.key(0), x, training=True)
    y = model.apply(variables, x, training=False)
    assert y.shape == x.shape
