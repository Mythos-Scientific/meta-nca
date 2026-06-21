import jax

from metanca.functional import flatten_params
from metanca.nn._conv_block import ConvBlock

CONV_LAYER_SPEC = [
    (6, (5, 5), (4, 4), (4, 4)),
    (12, (5, 5), (4, 4), (4, 4)),
    (25, (5, 5), (4, 4), (4, 4)),
]


def test_conv_block_instantiate():
    key = jax.random.key(0)
    data_key, param_key = jax.random.split(key, num=2)
    # imagenet dimensions
    x = jax.random.normal(data_key, shape=(4, 224, 224, 3))

    block = ConvBlock(layer_specs=CONV_LAYER_SPEC, padding="VALID", strides=1, use_bias=False)
    params = block.init(param_key, x)

    flattened_params = flatten_params(params["params"])

    filter_channels, kernel_sizes, *_ = zip(*CONV_LAYER_SPEC)

    input_channels = x.shape[-1]
    all_input_channels = (input_channels,) + filter_channels

    expected_shapes = []

    for input_channels, output_channels, kernel_size in zip(
        all_input_channels[:-1], all_input_channels[1:], kernel_sizes
    ):
        expected_shapes.append(kernel_size + (input_channels, output_channels))

    for (_, parameter), expected_shape in zip(flattened_params, expected_shapes):
        assert parameter.shape == expected_shape
