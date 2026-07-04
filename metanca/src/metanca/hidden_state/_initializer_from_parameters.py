import itertools
import re
from typing import Iterable, TypeVar

import jax
import jax.numpy as jnp

from ._hidden_state_initializer import HiddenStateInitializer, hidden_state_initializer

T = TypeVar("T")


def _unique_everseen(iterable: Iterable[T]) -> Iterable[T]:
    seen = set()
    for element in itertools.filterfalse(seen.__contains__, iterable):
        seen.add(element)
        yield element


def _cast_and_expand(shape: tuple[int, ...], rank: int) -> jax.Array:
    return jnp.asarray((1,) * (rank - len(shape)) + shape, dtype=int)


def get_layer_names_from_shapes(layer_shapes: dict) -> list[str]:
    """Extract unique layer names from a dictionary of layer shapes.

    Args:
        layer_shapes: Dictionary mapping parameter names (e.g., "Dense_0.kernel")
            to their shapes.

    Returns:
        List of unique layer names in order of first occurrence.
    """
    # Strip recognised Flax param-role suffixes so siblings (kernel/bias,
    # embedding, scale) collapse onto a single layer identity.
    return list(
        _unique_everseen(
            re.sub(r"\.(?:bias|kernel|embedding|scale)$", "", name) for name in layer_shapes
        )
    )


def create_unified_initializer(
    all_layer_shapes: dict,
    max_n_layers: int,
    d_neuron: int = 2,
    d_spatial: int = 2,
    d_layer: int = 2,
    n_spatial_dims: int = 0,
) -> tuple[HiddenStateInitializer, int]:
    """Create a unified hidden state initializer from shapes across multiple architectures.

    This function creates an initializer that can handle any architecture within
    the set of architectures whose shapes are provided. It computes the maximum
    dimensions needed across all architectures.

    Args:
        all_layer_shapes: Dictionary mapping parameter names to their maximum shapes
            across all architectures being considered.
        max_n_layers: Maximum number of layers across all architectures.
        d_neuron: Dimension of neuron positional encoding.
        d_spatial: Dimension of spatial positional encoding.
        d_layer: Dimension of layer positional encoding.
        n_spatial_dims: Number of spatial dimensions.

    Returns:
        Tuple of (initializer_function, hidden_dim).
    """
    # Find maximum dimensions across all primary param shapes (kernel/embedding/scale).
    _primary_suffixes = (".kernel", ".embedding", ".scale")
    primary_shapes = {
        k: v for k, v in all_layer_shapes.items() if any(k.endswith(s) for s in _primary_suffixes)
    }
    if not primary_shapes:
        primary_shapes = {k: v for k, v in all_layer_shapes.items() if ".bias" not in k}
    if not primary_shapes:
        raise ValueError("No primary param shapes found in all_layer_shapes")

    max_ndim = max(max(len(v) for v in primary_shapes.values()), n_spatial_dims + 2)

    all_kernel_shapes = jnp.stack(
        [_cast_and_expand(shape, max_ndim) for shape in primary_shapes.values()], axis=0
    )
    max_layer_dims = jnp.max(all_kernel_shapes, axis=0)

    initializer, hidden_dim = hidden_state_initializer(
        max_layer_dims, max_n_layers, d_layer=d_layer, d_neuron=d_neuron, d_spatial=d_spatial
    )
    return (initializer, hidden_dim)


def initializer_from_parameters(
    layer_shapes: dict,
    d_neuron: int = 2,
    d_spatial: int = 2,
    d_layer: int = 2,
    n_spatial_dims: int = 0,
) -> tuple[HiddenStateInitializer, int, list[str]]:
    layers = get_layer_names_from_shapes(layer_shapes)
    max_ndim = max(max(len(v) for v in layer_shapes.values()), n_spatial_dims + 2)

    _primary_roles = ("kernel", "embedding", "scale", "bias")

    def _primary_shape(layer: str) -> tuple[int, ...]:
        for role in _primary_roles:
            key = f"{layer}.{role}"
            if key in layer_shapes:
                return layer_shapes[key]
        raise KeyError(f"No recognized param role found for layer {layer}")

    all_kernel_shapes = jnp.stack(
        [_cast_and_expand(_primary_shape(layer), max_ndim) for layer in layers], axis=0
    )
    max_layer_dims = jnp.max(all_kernel_shapes, axis=0)
    initializer, hidden_dim = hidden_state_initializer(
        max_layer_dims, len(layers), d_layer=d_layer, d_neuron=d_neuron, d_spatial=d_spatial
    )
    return (initializer, hidden_dim, layers)
