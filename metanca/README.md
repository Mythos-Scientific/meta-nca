# metanca

`metanca` is the core MetaNCA library. It contains the data structures and neural modules that turn an ordinary Flax model into a parameter-space cellular automaton:

- a wrapped model becomes a `TaskNet`,
- each parameter tensor gets hidden state and positional encodings,
- parameter neighborhoods are compiled into forward/backward adjacency graphs, and
- a learnable local rule updates the TaskNet weights by reading only local parameter views.

This package does not own experiment orchestration, datasets, checkpointing, or Hydra configs. Those live in [`metanca_training/`](../metanca_training/).

## Public Surface

The main public exports are:

- `metanca.TaskNet`
- `metanca.update_tasknet`
- `metanca.nn.TaskNet`
- `metanca.nn.LocalRuleNet`
- `metanca.nn.LinearAttention`
- `metanca.nn.MultiLayerPerceptron`
- `metanca.nn.ConvMLP`
- `metanca.nn.ResNet`

The package also re-exports the `functional`, `neighbors`, `nn`, `strategies`, and `typing` subpackages at the top level. Treat the list above as the most common runtime entry points, not a complete export inventory.

At the object level, the intended flow is:

1. Build a `TaskNet` from a normal Flax model.
2. Initialize a `LocalRuleNet`.
3. Call `update_tasknet(...)` on the `TaskNet` parameter/state fields, then rebuild the runtime object with `tasknet.with_updated(...)` or the higher-level helpers in `metanca_training/`.

## Conceptual Model

### TaskNet

`TaskNet` in [`src/metanca/nn/_tasknet.py`](src/metanca/nn/_tasknet.py) is the central runtime object. It bundles:

- model parameters,
- per-parameter hidden states,
- per-parameter positional encodings,
- forward and backward adjacency metadata,
- layer ordering and dimensional metadata needed by the local rule.

`TaskNet.build(...)` and `TaskNet.build_many(...)`:

- initialize the wrapped Flax model,
- trace its JAX computation graph,
- derive parameter-neighborhood graphs,
- build hidden-state initializers and positional encodings, and
- return immutable `TaskNet` values that can be reset or updated.

Branch-specific details worth knowing:

- `TaskNet` is a frozen dataclass registered as a JAX pytree.
- The main convenience methods are `reset(...)`, `apply(...)`, `with_updated(...)`, `iter_param_names(...)`, and `get(...)`.
- `TaskNet.build_many(...)` is the shared-initializer path used to build multiple architectures with compatible hidden-state geometry.

One important implementation detail is the Conv-to-Dense boundary fix in [`_tasknet.py`](src/metanca/nn/_tasknet.py): neuron positional encodings are rearranged so flattened dense inputs preserve the intended spatial/channel neighborhood structure.

### Local Rule Update

The local update path lives in [`src/metanca/nn/_update_tasknet.py`](src/metanca/nn/_update_tasknet.py). For each parameter tensor it:

1. sample which parameter cells will be updated,
2. materialize forward/backward local views,
3. run the local rule network on those views, and
4. scatter the predicted deltas back into parameter and hidden-state tensors.

The execution loop is still centralized in one function, with neighborhood construction factored into [`src/metanca/nn/_get_neighbors.py`](src/metanca/nn/_get_neighbors.py). That split is the main seam for shape/alignment work on this branch.

### Neighborhood View Construction

[`src/metanca/nn/_get_neighbors.py`](src/metanca/nn/_get_neighbors.py) turns TaskNet adjacency metadata into forward and backward neighborhood views for one parameter at a time. It is responsible for:

- gathering focus and neighbor tensors from the parameter graph,
- aligning kernel and bias tensors into the neuron-view layout consumed by the local rule, and
- preserving the forward/backward slice metadata that `update_tasknet(...)` needs to reshape updates back into parameter space.

### From Paper Neighborhoods To Runtime Views

In the paper, a weight's forward neighborhood `Nf(w)` and backward neighborhood `Nb(w)` are defined by which downstream and upstream neurons or channels that weight touches. In the codebase, that same idea is compiled and materialized in three stages:

1. `TaskNet.build(...)` in [`src/metanca/nn/_tasknet.py`](src/metanca/nn/_tasknet.py) traces the wrapped Flax model to JAXPR, builds a compute graph, and asks [`src/metanca/neighbors/`](src/metanca/neighbors/) to derive forward and backward parameter-neighborhood graphs.
2. [`src/metanca/neighbors/_parameter_graph/_build_parameter_neighbor_graphs.py`](src/metanca/neighbors/_parameter_graph/_build_parameter_neighbor_graphs.py) walks outward from each parameter through the traced graph. Barrier ops such as `dot_general` and `conv_general_dilated` define real message-passing boundaries between parameters, while non-barrier ops such as `add` and `mul` keep parameters in the same local neighborhood.
3. [`src/metanca/neighbors/_convert_parameter_graph.py`](src/metanca/neighbors/_convert_parameter_graph.py) converts those graphs into the `TaskNet.adj` metadata stored on the runtime object.

The stored adjacency is intentionally compact. Rather than precomputing explicit neighbor indices for every scalar weight, `TaskNet.adj` is an [`AdjacencyDict`](src/metanca/typing/_adjacency_dict.py) keyed by focus parameter name, where each entry is a tuple of:

- `neighbor_name`,
- `message_fn`, and
- primitive-specific `message_fn_params`.

Those `message_fn_params` capture the geometry needed to reconstruct local views later, such as dense contraction axes or convolution input/output channel alignment. The corresponding slice-strategy layer in [`src/metanca/strategies/`](src/metanca/strategies/) turns that metadata into a [`MsgSliceInfo`](src/metanca/typing/_slice_info.py) for the focus tensor and the neighbor tensor, with `index_dim`, `stride`, and `slice_size` telling the runtime how to reinterpret each parameter as a neuron-oriented local neighborhood.

At update time, [`src/metanca/nn/_get_neighbors.py`](src/metanca/nn/_get_neighbors.py) applies those slice strategies and builds `NeuronView` tensors by swapping the aligned neuron axis to the front and flattening the remaining structure into the local feature axis. [`src/metanca/nn/_update_tasknet.py`](src/metanca/nn/_update_tasknet.py) then concatenates the forward and backward views into the tensors consumed by `LocalRuleNet`, runs the local rule, and inverts the reshape to scatter predicted deltas back into the original parameter layout. That is the concrete bridge from the paper's neighborhood language to the branch's actual neighbor-finding algorithm and data structures.

### Hidden State And Positional Encodings

[`src/metanca/hidden_state/_hidden_state_initializer.py`](src/metanca/hidden_state/_hidden_state_initializer.py) builds the hidden-state and positional-encoding channel layout from spatial, neuron, and layer encodings. `TaskNet.build(...)` uses that initializer family together with the Conv-to-Dense boundary fix in [`src/metanca/nn/_tasknet.py`](src/metanca/nn/_tasknet.py) so flattened dense inputs preserve the intended spatial/channel neighborhood structure.

## Core Neural Modules

### `LocalRuleNet`

[`src/metanca/nn/_local_rule_net.py`](src/metanca/nn/_local_rule_net.py) is the learnable local update rule. It consumes:

- forward and backward focus vectors,
- forward and backward neighbor vectors,
- positional encodings, and
- selected update masks.

It predicts:

- `delta_theta` for the parameter values, and
- `delta_hidden` for the hidden-state channels.

Internally, `LocalRuleNet` performs four conceptual phases inside one module:

- directional attention,
- backward-to-forward alignment,
- selected-cell perception assembly,
- local-rule head application.

This makes forward/backward fusion, sparse selection, and alignment experiments easier to isolate.

### `LinearAttention`

[`src/metanca/nn/_linear_attention.py`](src/metanca/nn/_linear_attention.py) provides the kernelized multi-head attention used inside the local rule. On this branch it is still implemented as one compact module, but the internal sequence is easy to follow:

- key/value input preparation,
- Q/K/V projection,
- rotary and mask application,
- KV summary formation,
- query application,
- output and optional post-FFN stack.

The refactor is meant to support kernel-level experiments without rewriting the whole module.

## Package Layout

- [`src/metanca/nn/`](src/metanca/nn/)
  Core neural modules, `TaskNet`, local-rule updates, and model definitions.
- [`src/metanca/neighbors/`](src/metanca/neighbors/)
  Conversion from traced compute graphs into parameter-neighborhood graphs.
- [`src/metanca/hidden_state/`](src/metanca/hidden_state/)
  Hidden-state and positional-encoding initialization logic.
- [`src/metanca/functional/`](src/metanca/functional/)
  Tensor reshaping, masking, flattening, and utility helpers used across the hot path.
- [`src/metanca/strategies/`](src/metanca/strategies/)
  Slice-strategy registry and dispatch for neighborhood view construction.
- [`src/metanca/typing/`](src/metanca/typing/)
  Shared type aliases and lightweight schema objects.

## Notes For Contributors

- `update_tasknet(...)` is the execution entry point; `get_neighbors(...)` is the main neighborhood-construction seam.
- If you need to change hidden-state or positional-encoding behavior, prefer doing it in `hidden_state/` and `TaskNet` construction rather than hard-coding shape rules into the update loop.
- `LocalRuleNet` and `LinearAttention` are the main experimentation surfaces for local-rule behavior and attention-kernel changes.
- If you want the training harness, benchmarks, or Hydra configs, start in [`metanca_training/README.md`](../metanca_training/README.md).
