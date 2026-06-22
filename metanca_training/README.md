# metanca_training

`metanca_training` is the experiment and training layer for the `metanca` library. It does not define the core neural cellular automata machinery itself; that lives in `metanca/`. Instead, this package is responsible for:

- composing experiments with Hydra,
- loading and batching datasets,
- instantiating ordinary Flax models that will be wrapped as `metanca.TaskNet`s,
- training a learnable local update rule that rewrites TaskNet parameters over time,
- validating that rule on held-out architectures,
- checkpointing and logging runs, and
- providing a standard backpropagation baseline for comparison.

The package is specialized. It is not a generic trainer in the style of Lightning or Keras. Its main job is to train a parameter-space local rule network that operates on `metanca`'s graph representation of another model's weights and hidden state.

## Conceptual Model

The training setup has three layers:

1. **Task architectures**
   Hydra model configs describe ordinary Flax modules such as `metanca.nn.MultiLayerPerceptron`, `metanca.nn.ConvMLP`, and `metanca.nn.ResNet`.
2. **TaskNets**
   `metanca.TaskNet.build_many(...)` converts those models into `TaskNet` objects with parameter tensors, hidden states, positional encodings, and adjacency graphs that describe local parameter neighborhoods.
3. **Local rule network**
   A separate learnable network, `metanca.nn.LocalRuleNet`, observes a local neighborhood around each parameter and predicts how that parameter and its hidden state should change.

Training optimizes the local rule network, not the task architecture weights directly. Each meta-training step resets the TaskNets, applies the local rule for some number of update steps, measures task loss on the resulting TaskNet parameters, and backpropagates through that unrolled process into the local rule parameters.

## Main Entry Points

### `scripts/train.py`

This is the primary MetaNCA training entry point. It:

- registers Hydra structured configs from [`src/metanca_training/_hydra_configs.py`](src/metanca_training/_hydra_configs.py),
- composes a config from `configs/`,
- loads one dataset,
- instantiates a list of training architectures plus one held-out test architecture with `hydra.utils.instantiate`,
- initializes the local rule network and optimizer, and
- runs `train_metanca(...)`.

The script expects direct execution, for example:

```bash
python metanca_training/scripts/train.py dataset=iris model=iris wandb=disabled
python metanca_training/scripts/train.py dataset=mnist_image model=mnist_conv wandb=disabled
```

`pyproject.toml` does not currently register console scripts, so the package is driven by these script files directly.
The `train.py` docstring also mentions `training=fast`, but no `fast` training preset is currently checked into `configs/training/`.

### `scripts/train_regular_net.py`

This is the comparison baseline. It trains the configured architectures directly with Adam and ordinary gradient descent instead of the MetaNCA local-rule mechanism. It uses the same dataset/config system and the same checkpointing helpers, which makes it the package's main apples-to-apples baseline path.

## Configuration System

Hydra is the public control surface for the package. The config schema is defined with dataclasses in [`src/metanca_training/_hydra_configs.py`](src/metanca_training/_hydra_configs.py), and the concrete YAML presets live in `configs/`.

### Root configs

- [`configs/config.yaml`](configs/config.yaml): MetaNCA training
- [`configs/config_regular_net.yaml`](configs/config_regular_net.yaml): regular-net baseline

### Config groups

- `dataset/`
  - `iris.yaml`
  - `mnist_image.yaml`
  - `mnist_flat.yaml`
  - `cifar100.yaml`
  - `imagenet.yaml`
- `model/`
  - MLP presets for Iris and MNIST
  - ConvMLP presets for MNIST, CIFAR-100, and ImageNet
  - ResNet presets for MNIST, CIFAR-100, and ImageNet
- `training/`
  - `default.yaml` for MetaNCA
  - `regular_net.yaml` for the baseline
- `positional_encoding/`
  - dimensions for layer, neuron, and spatial encodings
- `local_rule/`
  - hidden-layer widths for the local rule network
- `logging/`, `checkpoint/`, and `wandb/`
  - runtime frequency, checkpoint policy, and experiment logging

Implementation details worth knowing:

- Structured configs are registered before Hydra initializes, so config validation happens against the dataclass schema.
- The model configs describe raw Flax modules. They are not serialized `TaskNet`s.
- The dataset preset names in the tree are the actual Hydra group names. In practice it is safest to pass `dataset=...` explicitly because the checked-in root configs and some script docstrings still mention `dataset: mnist`, while the concrete dataset presets are `mnist_image` and `mnist_flat`.
- The checked-in MetaNCA training group currently contains only `training/default.yaml`; `training/regular_net.yaml` belongs to the baseline script.
- `training.num_metaepochs` is the outer optimization loop.
- With `training.update_step_scheduler_type=constant`, `training.num_epochs` is the fixed number of TaskNet local-rule update steps.
- With the checked-in `increment` scheduler, a fresh run starts at 1 update step and increases every `update_step_scheduler_rate` metaepochs, optionally capped by `update_step_scheduler_max_steps`. In that mode `training.num_epochs` is legacy naming, not the effective starting step count.

### Derived configuration

[`src/metanca_training/_config_factory.py`](src/metanca_training/_config_factory.py) computes two important runtime values:

- `hidden_state_dim = (#spatial_dims * d_spatial) + 2 * d_neuron + d_layer`
- `local_rule_arch = [3 + 3 * hidden_state_dim] + hidden_layers + [1 + hidden_state_dim]`

Those formulas determine the input/output size of the local rule network from the TaskNet encoding dimensions rather than hard-coding them in YAML.

## Data Pipeline

Dataset loading is centralized in [`src/metanca_training/data_utils.py`](src/metanca_training/data_utils.py).

Supported dataset paths:

- **Iris** via `sklearn.datasets.load_iris`, one-hot encoded and randomly split
- **MNIST** via OpenML, standardized with `StandardScaler`, then optionally reshaped into image form
- **CIFAR-100** via `tensorflow_datasets`, normalized with fixed channel statistics
- **ImageNet-style directory trees** via manual filesystem traversal plus a synset mapping file

Important implementation details:

- `prepare_batches(...)` converts arrays into `(n_batches, batch_size, ...)` form and pads only the last batch when necessary.
- For non-ImageNet datasets, `batch_size: 0` means "treat the whole split as one batch".
- Every batched dataset carries a boolean mask with shape `(n_batches, batch_size, 1)` so padded examples do not contribute to loss or accuracy.
- `masked_softmax_cross_entropy(...)` returns zero for an all-false mask, which avoids NaNs on fully padded slices.
- MNIST and CIFAR-100 are re-split inside the loader instead of using their canonical upstream train/validation partitions. CIFAR-100 also concatenates the upstream TFDS train and test sets before reshuffling.
- The active ImageNet loader reads `<data_dir>/train`, constructs prebatched arrays directly, and random-splits that tree into train/validation batches instead of using a separate validation directory.
- `promote_image_batch(...)` converts `uint8` image batches to `float32` in `[0, 1]` on demand. Pre-normalized float inputs are left unchanged, so effective normalization remains dataset-specific.
- The ImageNet loader writes directly into preallocated batched tensors and uses masks for incomplete final batches.
- The regular-net baseline includes layout handling so image arrays can be transposed from NCHW to NHWC when the configured model expects NHWC.

## MetaNCA Training Flow

The main runtime lives in [`src/metanca_training/_train_metanca.py`](src/metanca_training/_train_metanca.py).

### Initialization

`before_metanca_training(...)` performs the setup phase:

- computes the TaskNet hidden-state dimension from the input shape and positional encoding config,
- constructs the full local-rule MLP width list,
- builds one `TaskNet` per training architecture plus one held-out test `TaskNet`,
- initializes `metanca.nn.LocalRuleNet` with dummy focus/neighbor/positional tensors,
- creates an Optax Muon optimizer (`optax.contrib.muon`),
- restores checkpointed local-rule state if present, and
- constructs the update-step schedule.

Checkpointing uses Orbax. The checkpoint payload stores:

- local-rule parameters,
- optimizer state, and
- JSON metadata describing the optimizer type and keyword arguments.

If `checkpoint.save_top_n <= 0`, all checkpoints are preserved. Otherwise the manager keeps the best `N` according to the monitored metric.
In the current MetaNCA path, that monitored metric is training-set `accuracy`.

### Outer loop

`train_metanca(...)` runs an outer loop over `metaepoch`.

Each metaepoch:

1. updates the current number of TaskNet update steps from the configured scheduler,
2. resets every training `TaskNet` with a fresh random key,
3. extracts `(params, hidden_states, positional_encodings)` tuples from each TaskNet,
4. shuffles training batches,
5. applies one local-rule optimization step per batch, and
6. periodically runs validation and callbacks.

One subtle implementation detail matters if you extend the loop: the adapted TaskNet state returned by a batch step is used for metrics and callbacks, but it is not written back into the batch loop's `tasknet_data_list`. Within a metaepoch, batches therefore start from that metaepoch's freshly reset TaskNet state, while only the local-rule parameters persist across batches.

The scheduler currently supports:

- `constant`: keep a fixed number of TaskNet update steps
- `increment`: increase the number of update steps every `update_step_scheduler_rate` metaepochs, optionally capped by `update_step_scheduler_max_steps`

### Per-batch optimization

The per-batch MetaNCA step is implemented in [`src/metanca_training/_metanca_train_step.py`](src/metanca_training/_metanca_train_step.py) and [`src/metanca_training/_calculate_metanca_gradients.py`](src/metanca_training/_calculate_metanca_gradients.py).

For each batch:

- the package unrolls the local rule for `n_update_steps` with `jax.lax.scan`,
- computes task loss for each training architecture after the updates,
- differentiates each architecture contribution with respect to the local-rule parameters,
- clips and averages those per-architecture gradients,
- applies the optimizer update through a small helper, and
- returns the updated local-rule parameters plus the updated TaskNet data.

The staged split is now:

- [`calculate_metanca_gradients(...)`](src/metanca_training/_calculate_metanca_gradients.py) for the unroll/loss/grad path on single-device runs
- [`calculate_metanca_gradients_multi_gpu(...)`](src/metanca_training/_calculate_metanca_gradients.py) for architecture-parallel multi-GPU dispatch and gradient gathering
- [`_clip_gradients.py`](src/metanca_training/_clip_gradients.py) for gradient-norm clipping
- [`metanca_optimizer_step(...)`](src/metanca_training/_metanca_optimizer_step.py) for `optimizer.update(...)` plus `optax.apply_updates(...)`

The unrolled TaskNet update is shared between training and validation in [`src/metanca_training/_update_tasknet_n_steps.py`](src/metanca_training/_update_tasknet_n_steps.py). The scan body is wrapped with `jax.checkpoint`, which trades extra compute for lower memory pressure during backpropagation through the update trajectory.

### Multi-GPU behavior

The multi-device path is explicit rather than `pmap`-centric:

- tasknet state is staged to devices once per metaepoch,
- local-rule parameters are copied once per device,
- each architecture is dispatched to one GPU,
- raw per-architecture gradients are gathered back to the first device, and
- the same explicit optimizer-tail logic is reused to accumulate, average, clip, and optimize in one jitted step.

This design is optimized around architecture-parallel execution rather than splitting a single model across replicas.

## Research Artifacts

The repo root currently includes a small `autoresearch/` directory with research artifacts for this branch:

- [`autoresearch/run.log`](../autoresearch/run.log) and [`autoresearch/results.tsv`](../autoresearch/results.tsv): ad hoc research outputs checked into the working tree
- [`skills/read-metanca-paper/references/full_paper.pdf`](../skills/read-metanca-paper/references/full_paper.pdf): the embedded MetaNCA paper used by the repo-local skill

This branch does not currently ship a stable benchmark-script surface under `autoresearch/`, so the maintained entry points remain the training scripts and the test suite.

## Validation, Logging, and Callbacks

Validation is handled by [`src/metanca_training/_validation_step.py`](src/metanca_training/_validation_step.py).

The validation procedure is:

- reset a fresh validation `TaskNet`,
- apply the local rule for the current number of update steps once at the beginning,
- keep those updated parameters fixed for the validation pass, and
- compute loss and callback metrics over every validation batch.

`train_metanca(...)` logs two validation views:

- held-out test architecture on validation data
- average of the training architectures on validation data

Logging uses:

- standard Python logging for console output,
- Weights & Biases for scalar metrics, and
- disabled-mode `wandb.init(mode="disabled")` when logging is off so downstream logging code does not need to special-case a missing run.

The callback system in [`src/metanca_training/callbacks/`](src/metanca_training/callbacks/) is deliberately functional:

- callback objects are frozen dataclasses,
- state updates return new callback instances,
- `TrainingContext` carries the current read-only training state, and
- `CallbackRunner` orchestrates hook execution and merges `HookResult`s.

The built-in callbacks currently cover:

- batch accuracy computation across one or more TaskNets
- early stopping based on a monitored metric

The callback framework is broader than the current default wiring. On this branch, the default early-stopping callback in `train_metanca(...)` is effectively inert: it is instantiated with `monitor="val/loss"`, validation currently logs bare metric names such as `loss` and `accuracy`, and the loop never calls `on_epoch_end(...)`. Treat early stopping as framework functionality that still needs integration work, not as an active default safeguard.

## Regular-Net Baseline Path

The regular baseline is implemented in [`src/metanca_training/regular_net_functions.py`](src/metanca_training/regular_net_functions.py) and driven by [`scripts/train_regular_net.py`](scripts/train_regular_net.py).

This path exists to answer a different question: how well do the same architecture presets train under ordinary supervised learning without the MetaNCA local-rule mechanism?

Implementation details:

- architectures can be selected from the training set, the held-out test architecture, or both,
- models are initialized from the same Hydra model configs,
- training uses Adam by default,
- BatchNorm-capable models preserve `batch_stats` in a custom `RegularNetTrainState`,
- losses honor the same batch masks used by the MetaNCA path, and
- checkpoints are written per architecture under `regular_net_<label>/`.

## Package Layout

- [`src/metanca_training/_hydra_configs.py`](src/metanca_training/_hydra_configs.py): structured config schema
- [`src/metanca_training/_config_factory.py`](src/metanca_training/_config_factory.py): derived dimensions and local-rule architecture helpers
- [`src/metanca_training/data_utils.py`](src/metanca_training/data_utils.py): dataset loading and batch/mask preparation
- [`src/metanca_training/_train_metanca.py`](src/metanca_training/_train_metanca.py): main MetaNCA training loop
- [`src/metanca_training/_metanca_train_step.py`](src/metanca_training/_metanca_train_step.py): jitted batch update logic
- [`src/metanca_training/_calculate_metanca_gradients.py`](src/metanca_training/_calculate_metanca_gradients.py): loss/gradient accumulation across architectures
- [`src/metanca_training/_clip_gradients.py`](src/metanca_training/_clip_gradients.py): gradient clipping helpers
- [`src/metanca_training/_metanca_optimizer_step.py`](src/metanca_training/_metanca_optimizer_step.py): optimizer-update helper
- [`src/metanca_training/_validation_step.py`](src/metanca_training/_validation_step.py): held-out evaluation path
- [`src/metanca_training/_checkpointing.py`](src/metanca_training/_checkpointing.py): Orbax save/restore helpers
- [`src/metanca_training/callbacks/`](src/metanca_training/callbacks/): functional callback system
- [`src/metanca_training/regular_net_functions.py`](src/metanca_training/regular_net_functions.py): direct-training baseline
- [`src/metanca_training/init_functions.py`](src/metanca_training/init_functions.py): local-rule initialization plus older research helpers
- [`src/metanca_training/visualization_tools.py`](src/metanca_training/visualization_tools.py): exploratory analysis/visualization utilities

## Testing Surface

The test suite under `metanca_training/tests/` currently focuses on the behavior this README describes:

- config-derived dimensions,
- model-config instantiation across all checked-in model presets,
- callback semantics,
- regular-net batch-shape handling, and
- NaN safety for masked losses.

That coverage reflects the package's current priorities: config correctness, stable data layout, and making the training harness safe to run across multiple architecture families and partially padded batches.
