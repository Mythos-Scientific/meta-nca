import jax
import jax.numpy as jnp
import pytest

from metanca.nn.resnet._resnet import ResNet


@pytest.mark.parametrize(
    ("use_bn", "act"),
    [
        (False, "relu"),
        (True, "relu"),
        (False, "silu"),
        (True, "leaky_relu"),
    ],
)
def test_resnet_forward_shape(use_bn: bool, act: str):
    x = jnp.ones((4, 32, 32, 3))
    model = ResNet(num_classes=10, stage_blocks=(2, 2, 2), base_channels=16, use_bn=use_bn, act=act)
    variables = model.init(jax.random.key(0), x, training=True)
    y = model.apply(variables, x, training=False)
    assert y.shape == (4, 10)


def test_resnet_runs_with_identity_activation():
    x = jnp.ones((2, 32, 32, 3))
    model = ResNet(
        num_classes=5,
        stage_blocks=(1, 1, 1),
        base_channels=8,
        use_bn=True,
        act="identity",
    )
    variables = model.init(jax.random.key(0), x, training=True)
    y = model.apply(variables, x, training=False)
    assert y.shape == (2, 5)


def test_resnet_stem_uses_2d_conv_kernel():
    x = jnp.ones((1, 32, 32, 3))
    model = ResNet(num_classes=10, stage_blocks=(1, 1, 1), base_channels=16, use_bn=False)
    variables = model.init(jax.random.key(0), x, training=False)
    stem_kernel = variables["params"]["Conv_0"]["kernel"]
    assert stem_kernel.shape == (3, 3, 3, 16)
