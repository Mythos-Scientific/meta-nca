import functools
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal, Optional, Sequence

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from frozendict import frozendict

import metanca.functional as mfx
import metanca.hidden_state as mhx
import metanca.neighbors as mnx
from metanca.typing import AdjacencyDict

logger = logging.getLogger(__name__)

static_metadata = {"static": True}

nonzero_jit = jax.jit(jnp.nonzero, static_argnames="size")


def _fix_conv_dense_boundary_pe(
    layer_names: tuple[str, ...],
    layer_shapes: dict,
    hidden_states: dict,
    positional_encodings: dict,
    n_spatial_dims: int,
    d_spatial: int,
    d_neuron: int,
) -> None:
    """Fix neuron PE ordering at Conv->Dense boundaries for NHWC->NCHW compatibility.

    In NHWC format (used by Flax), flattening a conv output interleaves channels:
    flat_idx = h * W * C + w * C + ch. This causes the Dense input neuron indices
    to be scattered when grouped by conv output channel.

    This function reassigns the in_neuron PE of Dense params at Conv->Dense boundaries
    to use NCHW-equivalent indices, so that each conv output neuron sees Dense neighbors
    with contiguous neuron PEs (matching the spatial structure of the conv output).
    """
    in_neuron_start = n_spatial_dims * d_spatial

    for i in range(len(layer_names) - 1):
        curr_kernel_name = f"{layer_names[i]}.kernel"
        next_kernel_name = f"{layer_names[i + 1]}.kernel"

        # Layers without a ``.kernel`` (LM embeddings, RMSNorm scales, etc.) can
        # never form a Conv->Dense boundary; skip them silently.
        if not (
            mfx.nested_contains(curr_kernel_name, layer_shapes)
            and mfx.nested_contains(next_kernel_name, layer_shapes)
        ):
            continue

        curr_shape = mfx.nested_get(curr_kernel_name, layer_shapes)
        next_shape = mfx.nested_get(next_kernel_name, layer_shapes)

        # Detect Conv->Dense boundary: Conv has >2 dims (HWIO), Dense has 2 dims
        if len(curr_shape) > 2 and len(next_shape) == 2:
            n_channels = curr_shape[-1]  # HWIO: last dim = output channels
            input_features = next_shape[0]  # Dense: first dim = input features

            if input_features % n_channels != 0:
                logger.warning(
                    f"Conv->Dense boundary {curr_kernel_name}->{next_kernel_name}: "
                    f"input_features={input_features} not divisible by n_channels={n_channels}"
                )
                continue

            spatial_size = input_features // n_channels

            # Compute NHWC->NCHW permutation for neuron PE
            # NHWC: nhwc_i = spatial * n_channels + ch
            # NCHW: nchw_i = ch * spatial_size + spatial
            nhwc_i = jnp.arange(input_features)
            perm = (nhwc_i % n_channels) * spatial_size + nhwc_i // n_channels

            def _patch_neuron_pe(pe):
                in_neuron_pe = pe[:, 0:1, in_neuron_start : in_neuron_start + d_neuron]
                nchw_pe = in_neuron_pe[perm]  # (input_features, 1, d_neuron)
                return pe.at[:, :, in_neuron_start : in_neuron_start + d_neuron].set(
                    jnp.broadcast_to(
                        nchw_pe, pe[:, :, in_neuron_start : in_neuron_start + d_neuron].shape
                    )
                )

            # Patch both hidden_states and positional_encodings
            for state_dict in (hidden_states, positional_encodings):
                pe = mfx.nested_get(next_kernel_name, state_dict)
                mfx.nested_set(next_kernel_name, _patch_neuron_pe(pe), state_dict)

            logger.info(
                f"Fixed Conv->Dense neuron PE: {next_kernel_name} "
                f"(n_channels={n_channels}, spatial_size={spatial_size})"
            )


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["params", "hidden_states", "positional_encodings"],
    meta_fields=[
        "model",
        "adj",
        "layer_names",
        "hidden_dim",
        "input_shape",
        "hidden_state_initializer",
        "n_spatial_dims",
        "d_spatial",
        "d_neuron",
        "dummy_input_dtype",
    ],
)
@dataclass(frozen=True)
class TaskNet:
    input_shape: Sequence[int]
    hidden_state_initializer: mhx.HiddenStateInitializer
    model: nn.Module = field(compare=False, hash=False, metadata=static_metadata)
    params: dict = field(compare=False, hash=False)
    hidden_states: dict = field(compare=False, hash=False)
    positional_encodings: dict = field(compare=False, hash=False)
    adj: frozendict[Literal["fwd", "bwd"], AdjacencyDict] = field(metadata=static_metadata)
    layer_names: tuple[str] = field(metadata=static_metadata)
    hidden_dim: int = field(metadata=static_metadata)
    n_spatial_dims: int = field(metadata=static_metadata)
    d_spatial: int = field(metadata=static_metadata)
    d_neuron: int = field(metadata=static_metadata)
    dummy_input_dtype: Any = field(default=jnp.float32, metadata=static_metadata)

    def reset(self, rand_key: jax.random.PRNGKey) -> "TaskNet":
        model = self.model

        dummy_input = jnp.zeros((1, *self.input_shape), dtype=self.dummy_input_dtype)
        params = model.init(rand_key, dummy_input)

        hidden_state_dict = {}
        positional_encoding_dict = {}
        layer_shapes = jax.tree.map(lambda x: x.shape, params)

        primary_param_priority = ("kernel", "embedding", "scale", "bias")
        for layer_idx, layer in enumerate(self.layer_names):
            layer_param_dict = mfx.nested_get(layer, layer_shapes)
            available_param_roles = tuple(layer_param_dict.keys())
            primary_role = next(
                (r for r in primary_param_priority if r in available_param_roles),
                available_param_roles[0],
            )
            primary_shape = layer_param_dict[primary_role]
            biased = "bias" in available_param_roles

            kernel_state, bias_state = self.hidden_state_initializer(
                primary_shape, layer_idx, bias=biased
            )

            for role in available_param_roles:
                if role == "bias" and bias_state is not None:
                    state = bias_state
                elif role == primary_role:
                    state = kernel_state
                else:
                    state, _ = self.hidden_state_initializer(
                        layer_param_dict[role], layer_idx, bias=False
                    )
                mfx.nested_set(f"{layer}.{role}", state, hidden_state_dict, delimiter=".")
                mfx.nested_set(f"{layer}.{role}", state, positional_encoding_dict, delimiter=".")

        _fix_conv_dense_boundary_pe(
            layer_names=self.layer_names,
            layer_shapes=layer_shapes,
            hidden_states=hidden_state_dict,
            positional_encodings=positional_encoding_dict,
            n_spatial_dims=self.n_spatial_dims,
            d_spatial=self.d_spatial,
            d_neuron=self.d_neuron,
        )

        return self.with_updated(
            params=params,
            hidden_states=hidden_state_dict,
            positional_encodings=positional_encoding_dict,
        )

    def apply(self, x: jax.Array) -> jax.Array:
        return self.model.apply(self.params, x)

    def with_updated(self, **kwargs) -> "TaskNet":
        self_dict = asdict(self)
        self_dict.update(kwargs)
        _ = self_dict.pop("model")
        return TaskNet(model=self.model, **self_dict)

    @classmethod
    def build(
        cls,
        model: nn.Module,
        input_shape: chex.Shape,
        key: chex.PRNGKey,
        bias_constant: float = -10.0,
        n_spatial_dims: int = 0,
        d_neuron: int = 2,
        d_spatial: int = 2,
        d_layer: int = 2,
        shared_initializer: Optional[tuple[mhx.HiddenStateInitializer, int]] = None,
        dummy_input_dtype: jnp.dtype = jnp.float32,
    ) -> "TaskNet":
        """Build a TaskNet from an architecture specification.

        Args:
            arch: Tuple of (conv_layer_spec, mlp_layer_spec).
            input_shape: Shape of input data.
            key: Random key for initialization.
            conv_use_bias: Whether to use bias in conv layers.
            mlp_use_bias: Whether to use bias in MLP layers.
            bias_constant: Constant value for bias hidden state initialization.
            n_spatial_dims: Number of spatial dimensions.
            d_neuron: Dimension of neuron positional encoding.
            d_spatial: Dimension of spatial positional encoding.
            d_layer: Dimension of layer positional encoding.
            shared_initializer: Optional tuple of (initializer_fn, hidden_dim) to use
                instead of creating a new one. Used by build_many to ensure all
                TaskNets share the same initializer.
            dummy_input_dtype: dtype of the probe tensor used to run ``model.init``.
                Image models accept the float32 default; token-id models (e.g.
                ``TinyCausalLM``) need an integer dtype like ``jnp.int32``.

        Returns:
            A new TaskNet instance.
        """
        dummy_input = jnp.zeros((1, *input_shape), dtype=dummy_input_dtype)
        params = model.init(key, dummy_input)

        def forward_fn(*args):
            return model.apply(params, *args)

        closed = jax.make_jaxpr(forward_fn)(dummy_input)
        jaxpr = closed.jaxpr
        compute_graph = mnx.build_compute_graph(jaxpr)
        flattened = mfx.flatten_params(params)
        # Models that do work like ``jnp.arange(seq_len)`` inside ``apply`` capture
        # extra non-parameter constants alongside the closed-over params. Identity-
        # match constvars against the flattened-param arrays via ``closed.literals``
        # so we keep only the real parameters and don't mis-pair names with vars.
        param_id_to_name = {id(param): name for name, param in flattened}
        param_id_to_value = {id(param): param for _, param in flattened}
        names_vars_and_params = [
            (param_id_to_name[id(lit)], var, param_id_to_value[id(lit)])
            for var, lit in zip(jaxpr.constvars, closed.literals)
            if id(lit) in param_id_to_name
        ]

        fwd, bwd = mnx.build_parameter_neighbor_graphs(
            compute_graph,
            barrier_ops=set(mnx.get_barrier_ops()),
            nonbarrier_ops=set(mnx.get_nonbarrier_ops()),
        )
        fwd_neighbor_data = mnx.convert_parameter_graph(
            fwd,
            names_vars_and_params,
        )
        bwd_neighbor_data = mnx.convert_parameter_graph_backward(
            bwd,
            names_vars_and_params,
        )

        layer_shapes = jax.tree.map(lambda x: x.shape, params)
        flattened_layer_shapes = {
            name: mfx.nested_get(name, layer_shapes) for name, _, _ in names_vars_and_params
        }

        # Extract layer names from flattened shapes
        layer_names = mhx.get_layer_names_from_shapes(flattened_layer_shapes)

        if shared_initializer is not None:
            # Use the provided shared initializer
            hidden_initializer, hidden_dim = shared_initializer
        else:
            # Create a new initializer specific to this architecture
            hidden_initializer, hidden_dim, layer_names = mhx.initializer_from_parameters(
                flattened_layer_shapes,
                d_spatial=d_spatial,
                d_layer=d_layer,
                d_neuron=d_neuron,
                n_spatial_dims=n_spatial_dims,
            )

        hidden_initializer_with_bias = functools.partial(
            hidden_initializer, bias_constant=bias_constant
        )
        hidden_state_dict = {}
        positional_encoding_dict = {}

        # Pick the primary param of each layer to drive the hidden-state shape.
        # Order matches Flax conventions across image/text models.
        primary_param_priority = ("kernel", "embedding", "scale", "bias")

        for layer_idx, layer in enumerate(layer_names):
            layer_param_dict = mfx.nested_get(layer, layer_shapes)
            available_param_roles = tuple(layer_param_dict.keys())
            primary_role = next(
                (r for r in primary_param_priority if r in available_param_roles),
                available_param_roles[0],
            )
            primary_shape = layer_param_dict[primary_role]
            biased = "bias" in available_param_roles

            kernel_state, bias_state = hidden_initializer_with_bias(
                primary_shape, layer_idx, bias=biased
            )

            for role in available_param_roles:
                if role == "bias" and bias_state is not None:
                    state = bias_state
                elif role == primary_role:
                    state = kernel_state
                else:
                    # Other params on this layer (e.g. a separate scale alongside a kernel)
                    # get their own per-shape init.
                    state, _ = hidden_initializer_with_bias(
                        layer_param_dict[role], layer_idx, bias=False
                    )
                mfx.nested_set(f"{layer}.{role}", state, hidden_state_dict, delimiter=".")
                mfx.nested_set(f"{layer}.{role}", state, positional_encoding_dict, delimiter=".")

        _fix_conv_dense_boundary_pe(
            layer_names=layer_names,
            layer_shapes=layer_shapes,
            hidden_states=hidden_state_dict,
            positional_encodings=positional_encoding_dict,
            n_spatial_dims=n_spatial_dims,
            d_spatial=d_spatial,
            d_neuron=d_neuron,
        )

        return cls(
            input_shape=input_shape,
            hidden_state_initializer=hidden_initializer_with_bias,
            model=model,
            params=params,
            hidden_states=hidden_state_dict,
            positional_encodings=positional_encoding_dict,
            adj=mfx.toggle_statics(
                {"fwd": fwd_neighbor_data, "bwd": bwd_neighbor_data}, freeze=True
            ),
            layer_names=mfx.toggle_statics(layer_names, freeze=True),
            hidden_dim=hidden_dim,
            n_spatial_dims=n_spatial_dims,
            d_spatial=d_spatial,
            d_neuron=d_neuron,
            dummy_input_dtype=dummy_input_dtype,
        )

    @classmethod
    def build_many(
        cls,
        models: list[nn.Module],
        input_shapes: list[chex.Shape],
        key: chex.PRNGKey,
        d_neuron: int = 2,
        d_layer: int = 2,
        d_spatial: int = 2,
        shared_initializer: Optional[tuple[mhx.HiddenStateInitializer, int]] = None,
        dummy_input_dtype: jnp.dtype = jnp.float32,
    ) -> list["TaskNet"]:
        """Build multiple TaskNets with a shared hidden state initializer.

        init_keys = jax.random.split(key, len(models))
        All TaskNets built by this method share the same hidden state initializer,
        which is computed based on the maximum dimensions across ALL architectures.
        This ensures consistent positional encodings and hidden states for
        generalization across architectures.

        Args:
            specs: List of (conv_layer_spec, mlp_layer_spec) tuples.
            input_shapes: List of input shapes corresponding to each spec.
            key: Random key for initialization.
            conv_use_bias: Whether to use bias in conv layers.
            mlp_use_bias: Whether to use bias in MLP layers.
            d_neuron: Dimension of neuron positional encoding.
            d_layer: Dimension of layer positional encoding.
            d_spatial: Dimension of spatial positional encoding.
        Returns:
            List of TaskNet instances, all sharing the same initializer.
        """
        n_spatial_dims = max(map(len, input_shapes)) - 1  # 1 would be the channel dim

        if shared_initializer is None:
            # First pass: collect all layer shapes across all architectures to compute
            # unified initializer dimensions
            all_layer_shapes = {}
            max_n_layers = 0
            probe_key, key = jax.random.split(key)
            probe_keys = jax.random.split(probe_key, len(models))

            for model, input_shape, probe_key in zip(models, input_shapes, probe_keys):
                dummy_input = jnp.zeros((1, *input_shape), dtype=dummy_input_dtype)
                params = model.init(probe_key, dummy_input)
                flattened = mfx.flatten_params(params)
                layer_shapes = jax.tree.map(lambda x: x.shape, params)

                for name, _ in flattened:
                    shape = mfx.nested_get(name, layer_shapes)
                    # Track the maximum shape for each dimension position
                    if name not in all_layer_shapes:
                        all_layer_shapes[name] = shape
                    else:
                        # Take element-wise max of shapes
                        existing = all_layer_shapes[name]
                        all_layer_shapes[name] = tuple(max(e, s) for e, s in zip(existing, shape))

                # Count layers for this architecture
                layer_names = mhx.get_layer_names_from_shapes(
                    {name: mfx.nested_get(name, layer_shapes) for name, _ in flattened}
                )
                max_n_layers = max(max_n_layers, len(layer_names))

            # Create a unified initializer using maximum dimensions across all architectures
            unified_initializer, unified_hidden_dim = mhx.create_unified_initializer(
                all_layer_shapes,
                max_n_layers=max_n_layers,
                d_spatial=d_spatial,
                d_layer=d_layer,
                d_neuron=d_neuron,
                n_spatial_dims=n_spatial_dims,
            )

            logger.info(
                f"Created unified initializer: hidden_dim={unified_hidden_dim}, "
                f"max_n_layers={max_n_layers}, n_spatial_dims={n_spatial_dims}"
            )
            shared_initializer = (unified_initializer, unified_hidden_dim)

        # Second pass: build TaskNets with shared initializer
        init_keys = jax.random.split(key, len(models))
        build_fn = functools.partial(
            cls.build,
            n_spatial_dims=n_spatial_dims,
            d_neuron=d_neuron,
            d_spatial=d_spatial,
            d_layer=d_layer,
            shared_initializer=shared_initializer,
            dummy_input_dtype=dummy_input_dtype,
        )

        return [
            build_fn(model=model, input_shape=input_shape, key=init_key)
            for (model, input_shape, init_key) in zip(models, input_shapes, init_keys)
        ]

    def iter_param_names(self) -> Iterable[str]:
        # Include all Flax param roles we recognise, not just kernel/bias —
        # otherwise embedding/scale params get silently dropped from updates.
        for layer_name in self.layer_names:
            for param_type in ("bias", "kernel", "embedding", "scale"):
                param_name = f"{layer_name}.{param_type}"
                if mfx.nested_contains(param_name, self.params):
                    yield param_name

    def get(
        self, name: str, category: Literal["hidden_state", "positional_encoding", "param"]
    ) -> jax.Array:
        return mfx.nested_get(
            name,
            {
                "hidden_state": self.hidden_states,
                "positional_encoding": self.positional_encodings,
                "param": self.params,
            }[category],
        )
