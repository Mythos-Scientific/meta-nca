import math

import jax
import jax.numpy as jnp

from metanca.functional import flatten_params
from metanca.nn._conv_mlp import ConvLayerSpec, ConvMLP, Padding

CONV_LAYER_SPEC = [
    (3, (5, 5), (4, 4), (4, 4)),
    (4, (5, 5), (4, 4), (4, 4)),
    (5, (5, 5), (4, 4), (4, 4)),
]

MLP_LAYER_SPEC = [10, 20, 100]


def conv_output_dim(size: int, kernel: int, stride: int, padding: Padding) -> int:

    match padding:
        case "VALID":
            return (size - kernel) // stride + 1
        case "SAME" | "CIRCULAR":
            return math.ceil(size / stride)
        case _:
            raise ValueError(f"Unknown padding {padding}")


def pool_output_dim(size: int, window: int, stride: int) -> int:
    return (size - window) // stride + 1


def convblock_output_shape(
    input_hw: tuple[int, int],
    layer_specs: ConvLayerSpec,
    conv_padding: Padding = "VALID",
    conv_stride: int = 1,
) -> tuple[int, int]:
    H, W = input_hw
    for _, kernel_size, pool_size, pool_strides in layer_specs:
        Kh, Kw = kernel_size
        Ph, Pw = pool_size
        Uh, Uw = pool_strides

        # conv
        H = conv_output_dim(H, Kh, conv_stride, conv_padding)
        W = conv_output_dim(W, Kw, conv_stride, conv_padding)

        # pool
        H = pool_output_dim(H, Ph, Uh)
        W = pool_output_dim(W, Pw, Uw)

    return (H, W)


def test_conv_mlp_instantiate():
    key = jax.random.key(0)
    data_key, param_key = jax.random.split(key, num=2)
    # imagenet dimensions
    x = jax.random.normal(data_key, shape=(1, 224, 224, 3))

    model = ConvMLP(
        CONV_LAYER_SPEC,
        MLP_LAYER_SPEC,
        mlp_use_bias=True,
        conv_use_bias=True,
        padding="VALID",
        strides=1,
    )
    params = model.init(param_key, x)

    flattened_params = flatten_params(params["params"])

    filter_channels, kernel_sizes, *_ = zip(*CONV_LAYER_SPEC)

    input_channels = x.shape[-1]
    all_input_channels = (input_channels,) + filter_channels

    expected_shapes = {}

    cnn_shape_iterable = list(zip(all_input_channels[:-1], all_input_channels[1:], kernel_sizes))

    for i, (input_channels, output_channels, kernel_size) in enumerate(cnn_shape_iterable):
        param_shape = kernel_size + (input_channels, output_channels)
        layer_name = f"ConvBlock_0.Conv_{i}"
        expected_shapes[f"{layer_name}.kernel"] = param_shape
        expected_shapes[f"{layer_name}.bias"] = (output_channels,)

    _, final_output_channels, *_ = cnn_shape_iterable[-1]

    cnn_output_shape = convblock_output_shape(
        (224, 224), CONV_LAYER_SPEC, conv_padding="VALID", conv_stride=1
    ) + (final_output_channels,)
    first_mlp_input_shape = int(jnp.prod(jnp.asarray(cnn_output_shape)))

    mlp_shape_iterable = zip([first_mlp_input_shape] + MLP_LAYER_SPEC[:-1], MLP_LAYER_SPEC)
    for i, (input_shape, output_shape) in enumerate(mlp_shape_iterable):
        layer_name = f"MultiLayerPerceptron_0.Dense_{i}"
        expected_shapes[f"{layer_name}.kernel"] = (input_shape, output_shape)
        expected_shapes[f"{layer_name}.bias"] = (output_shape,)

    flattened_param_shape_dict = jax.tree.map(lambda x: x.shape, dict(flattened_params))

    for name, shape in flattened_param_shape_dict.items():
        assert expected_shapes[name] == shape
