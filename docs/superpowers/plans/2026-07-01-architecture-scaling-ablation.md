# Architecture-Scaling Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how MetaNCA's architecture generalization scales with the number of training architectures T, via two ablations (fixed-depth D=5 and varying-depth D=2..5) over a grid of non-increasing power-of-two-width MLPs on Fashion-MNIST.

**Architecture:** Add a pure `scaling/arch_grid.py` (enumerate/split/sample architectures) and `scaling/evaluate_pool.py` (post-training per-arch evaluation). Add a Fashion-MNIST loader. Make `train_metanca` return its final local-rule params + training vars so a driver script can meta-train on T archs and evaluate the resulting rule on all train + held-out val archs in the same process, writing per-arch rows to JSONL. Adam baselines and plotting are standalone scripts.

**Tech Stack:** Python 3.12, JAX (CUDA on GB10/aarch64), Flax, Optax, Hydra/OmegaConf, Orbax, tensorflow-datasets, matplotlib, pytest, uv.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-01-architecture-scaling-ablation-design.md`.
- Branch: `architecture-scaling-ablation`.
- Hidden widths ∈ **{32, 64, 128, 256, 512}** (powers of two), **non-increasing** toward output; constant-width allowed.
- Output classes = **10** (Fashion-MNIST). Arch instantiated as `MultiLayerPerceptron(layer_specs=[*widths, 10], activation="leaky_relu", final_layer_activation="identity", use_bias=True)`.
- Fixed-depth grid (D=5) = **126** archs; hold out **V=25**, pool=**101**. Varying-depth grid (D=2..5) = **246** archs; hold out **V=49**, pool=**197**. Held-out V fixed within an ablation (seeded).
- Sweep **T ∈ {1, 5, 10, 50, 100}**, **3 independent replicate runs** per T (fresh random T-subset per rep). No trimming of reps.
- Training: **12,000 metaepochs**, update-step schedule `increment`, `rate=1000`, `max_steps=10`. Early stopping **disabled**. Sample pooling **off**.
- Evaluation: **final** local-rule params, **10** update steps, **5** random inits/arch, record val **loss** and val **accuracy** for all train + all val archs.
- Env var on GB10 (unified memory): set `XLA_PYTHON_CLIENT_PREALLOCATE=false` for all runs.
- Results dir: `results/`. Every task ends with a commit.

Key existing signatures this plan depends on (already in the repo — do not redefine):
- `metanca.TaskNet.build(model=, input_shape=, key=, n_spatial_dims=, d_neuron=, d_spatial=, d_layer=, shared_initializer=)` returns a `TaskNet` with attributes `.hidden_state_initializer`, `.hidden_dim`, `.model`, `.adj`, `.params`, `.hidden_states`, `.positional_encodings`, `.iter_param_names()`, `.reset(key)`.
- `metanca_training._train_metanca.before_metanca_training(cfg, *, models, test_model, rand_key) -> TrainingVars` (has `.tasknets`, `.test_tasknet`, `.local_rule_net_apply`, `.local_rule_params`, `.hidden_dim`, `.n_update_steps`).
- `metanca_training._validation_step.metanca_validation_step(val_batches, local_rule_net_apply, local_rule_params, rand_key, val_tasknet, n_update_steps, hidden_dim, prop_cells_updated, weight_transformer_dropout, callback_runner, n_spatial_dims) -> (dict, CallbackRunner)` — returned dict has `"loss"` and `"accuracy"`.
- `metanca_training.callbacks.CallbackRunner.create([create_accuracy_callback()])`.
- `metanca.nn.MultiLayerPerceptron(layer_specs=, activation=, final_layer_activation=, use_bias=)`.
- `metanca_training.data_utils.prepare_batches(X, y, train_inds, val_inds, batch_size=)`.

---

### Task 1: uv environment + verification

**Files:**
- Create: `README_scaling.md` (short env/run notes for this study; also record whether stock `jax[cuda12]` worked or an NVIDIA JAX build was needed).

**Interfaces:**
- Produces: a working venv at `.venv/` where `import jax; jax.devices()` shows the GB10 GPU and the existing test suite passes.

- [ ] **Step 1: Create venv and install**

```bash
cd /home/dan/Projects/meta-nca
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.in -r metanca/requirements.in -r metanca_training/requirements.in
uv pip install -e ./metanca -e ./metanca_training
uv pip install tensorflow-cpu
```

- [ ] **Step 2: Verify JAX sees the GPU**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false python -c "import jax; print(jax.devices())"`
Expected: a non-empty device list including a CUDA/GB10 device.
If it fails or shows only CPU: stock `jax[cuda12]` wheels likely do not cover aarch64/Blackwell — install JAX from NVIDIA's build/container (document what worked in `README_scaling.md`). **Do not proceed until a GPU device appears.**

- [ ] **Step 3: Run the existing test suite**

Run: `pytest metanca/ metanca_training/ -q`
Expected: PASS (baseline green before changes).

- [ ] **Step 4: Commit**

```bash
git add README_scaling.md
git commit -m "chore: set up uv env for scaling study; document GB10 JAX setup"
```

---

### Task 2: Architecture grid (pure logic)

**Files:**
- Create: `metanca_training/src/metanca_training/scaling/__init__.py`
- Create: `metanca_training/src/metanca_training/scaling/arch_grid.py`
- Test: `metanca_training/tests/metanca_training/scaling/__init__.py`, `metanca_training/tests/metanca_training/scaling/test__arch_grid.py`

**Interfaces:**
- Produces (used by Tasks 6, 7, 9, 10):
  - `WIDTHS: tuple[int, ...] = (32, 64, 128, 256, 512)`
  - `enumerate_arch_widths(depths: Sequence[int], widths: Sequence[int] = WIDTHS) -> list[tuple[int, ...]]` — all non-increasing width tuples of each depth.
  - `arch_id(widths: Sequence[int]) -> str` → e.g. `"d3_512-128-64"`.
  - `arch_layer_specs(widths: Sequence[int], n_classes: int) -> list[int]` → `[*widths, n_classes]`.
  - `build_mlp(widths: Sequence[int], n_classes: int, use_bias: bool = True) -> MultiLayerPerceptron`.
  - `build_grid(kind: str) -> list[tuple[int, ...]]` — `kind` ∈ `{"fixed5", "varying"}`.
  - `split_grid(grid: Sequence[tuple[int, ...]], n_val: int, seed: int) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]` → `(pool, val)`.
  - `sample_subset(pool: Sequence[tuple[int, ...]], t: int, seed: int) -> list[tuple[int, ...]]`.
  - `N_VAL: dict[str, int] = {"fixed5": 25, "varying": 49}`.

- [ ] **Step 1: Write the failing tests**

```python
# metanca_training/tests/metanca_training/scaling/test__arch_grid.py
from metanca_training.scaling.arch_grid import (
    WIDTHS, N_VAL, enumerate_arch_widths, arch_id, arch_layer_specs,
    build_mlp, build_grid, split_grid, sample_subset,
)


def test_enumerate_counts_per_depth():
    assert len(enumerate_arch_widths([2])) == 15
    assert len(enumerate_arch_widths([3])) == 35
    assert len(enumerate_arch_widths([4])) == 70
    assert len(enumerate_arch_widths([5])) == 126


def test_enumerate_non_increasing_and_valid_widths():
    for w in enumerate_arch_widths([2, 3, 4, 5]):
        assert all(a >= b for a, b in zip(w, w[1:])), w
        assert all(x in WIDTHS for x in w), w


def test_enumerate_no_duplicates():
    archs = enumerate_arch_widths([2, 3, 4, 5])
    assert len(archs) == len(set(archs))


def test_build_grid_sizes():
    assert len(build_grid("fixed5")) == 126
    assert len(build_grid("varying")) == 246
    assert all(len(w) == 5 for w in build_grid("fixed5"))
    assert {len(w) for w in build_grid("varying")} == {2, 3, 4, 5}


def test_arch_id_and_layer_specs():
    assert arch_id((512, 128, 64)) == "d3_512-128-64"
    assert arch_layer_specs((512, 128, 64), 10) == [512, 128, 64, 10]


def test_split_grid_deterministic_disjoint():
    grid = build_grid("fixed5")
    pool_a, val_a = split_grid(grid, N_VAL["fixed5"], seed=0)
    pool_b, val_b = split_grid(grid, N_VAL["fixed5"], seed=0)
    assert val_a == val_b and pool_a == pool_b            # deterministic
    assert len(val_a) == 25 and len(pool_a) == 101
    assert set(pool_a).isdisjoint(set(val_a))
    assert set(pool_a) | set(val_a) == set(grid)
    _, val_c = split_grid(grid, N_VAL["fixed5"], seed=1)
    assert val_c != val_a                                 # seed changes split


def test_sample_subset_size_and_membership():
    pool, _ = split_grid(build_grid("fixed5"), N_VAL["fixed5"], seed=0)
    sub = sample_subset(pool, 5, seed=3)
    assert len(sub) == 5 and set(sub).issubset(set(pool))
    assert sample_subset(pool, 5, seed=3) == sub          # deterministic


def test_build_mlp_forward_shape():
    import jax, jax.numpy as jnp
    model = build_mlp((32, 32), n_classes=10)
    params = model.init(jax.random.key(0), jnp.zeros((4, 784)))
    out = model.apply(params, jnp.zeros((4, 784)))
    assert out.shape == (4, 10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest metanca_training/tests/metanca_training/scaling/test__arch_grid.py -q`
Expected: FAIL (module `metanca_training.scaling.arch_grid` not found).

- [ ] **Step 3: Implement the module**

```python
# metanca_training/src/metanca_training/scaling/arch_grid.py
"""Deterministic architecture grid for the scaling ablation.

An architecture is a tuple of hidden-layer widths, non-increasing, drawn from
powers of two in {32,64,128,256,512}. The output layer (n_classes) is appended
only at instantiation time.
"""

from itertools import combinations_with_replacement
from typing import Sequence

import jax
import numpy as np

from metanca.nn import MultiLayerPerceptron

WIDTHS: tuple[int, ...] = (32, 64, 128, 256, 512)

# Held-out validation-architecture counts (~20% of each grid).
N_VAL: dict[str, int] = {"fixed5": 25, "varying": 49}

_GRID_DEPTHS: dict[str, list[int]] = {"fixed5": [5], "varying": [2, 3, 4, 5]}


def enumerate_arch_widths(
    depths: Sequence[int], widths: Sequence[int] = WIDTHS
) -> list[tuple[int, ...]]:
    """All non-increasing width tuples of the given depths.

    combinations_with_replacement over descending-sorted widths yields exactly
    the non-increasing tuples, one per multiset (no duplicates).
    """
    desc = tuple(sorted(widths, reverse=True))
    archs: list[tuple[int, ...]] = []
    for d in depths:
        archs.extend(combinations_with_replacement(desc, d))
    return archs


def build_grid(kind: str) -> list[tuple[int, ...]]:
    if kind not in _GRID_DEPTHS:
        raise ValueError(f"unknown grid kind: {kind!r}")
    return enumerate_arch_widths(_GRID_DEPTHS[kind])


def arch_id(widths: Sequence[int]) -> str:
    return f"d{len(widths)}_" + "-".join(str(w) for w in widths)


def arch_layer_specs(widths: Sequence[int], n_classes: int) -> list[int]:
    return [*widths, n_classes]


def build_mlp(
    widths: Sequence[int], n_classes: int, use_bias: bool = True
) -> MultiLayerPerceptron:
    return MultiLayerPerceptron(
        layer_specs=arch_layer_specs(widths, n_classes),
        activation="leaky_relu",
        final_layer_activation="identity",
        use_bias=use_bias,
    )


def _shuffled(grid: Sequence[tuple[int, ...]], seed: int) -> list[tuple[int, ...]]:
    order = np.array(jax.random.permutation(jax.random.key(seed), len(grid)))
    return [tuple(grid[i]) for i in order]


def split_grid(
    grid: Sequence[tuple[int, ...]], n_val: int, seed: int
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Return (pool, val). Deterministic in seed; disjoint; covers the grid."""
    shuffled = _shuffled(grid, seed)
    val = shuffled[:n_val]
    pool = shuffled[n_val:]
    return pool, val


def sample_subset(
    pool: Sequence[tuple[int, ...]], t: int, seed: int
) -> list[tuple[int, ...]]:
    if t > len(pool):
        raise ValueError(f"cannot sample T={t} from pool of {len(pool)}")
    return _shuffled(pool, seed)[:t]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest metanca_training/tests/metanca_training/scaling/test__arch_grid.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add metanca_training/src/metanca_training/scaling/ metanca_training/tests/metanca_training/scaling/
git commit -m "feat: architecture grid enumeration/split/sample for scaling ablation"
```

---

### Task 3: Fashion-MNIST dataset

**Files:**
- Modify: `metanca_training/src/metanca_training/data_utils.py` (add `_prepare_fashion_mnist_arrays`, `get_fashion_mnist_datasets`).
- Create: `metanca_training/configs/dataset/fashion_mnist.yaml`
- Modify: `metanca_training/scripts/train.py:load_dataset` and `metanca_training/scripts/train_regular_net.py:load_dataset` (add `fashion_mnist` branch).
- Test: `metanca_training/tests/metanca_training/test__fashion_mnist_prepare.py`

**Interfaces:**
- Produces: `get_fashion_mnist_datasets(rand_key) -> (X, y, train_inds, val_inds)` with `X` float32 `(N, 784)` standardized, `y` one-hot `(N, 10)`, 80/20 split. Pure helper `_prepare_fashion_mnist_arrays(images_uint8, labels_int, rand_key) -> (X, y, train_inds, val_inds)`.

- [ ] **Step 1: Write the failing test** (pure transform only — no network/tfds in the test)

```python
# metanca_training/tests/metanca_training/test__fashion_mnist_prepare.py
import jax
import numpy as np

from metanca_training.data_utils import _prepare_fashion_mnist_arrays


def test_prepare_fashion_mnist_shapes_and_split():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(100, 28, 28), dtype=np.uint8)
    labels = rng.integers(0, 10, size=(100,), dtype=np.int32)

    X, y, train_inds, val_inds = _prepare_fashion_mnist_arrays(
        images, labels, jax.random.key(0)
    )

    assert X.shape == (100, 784) and X.dtype == np.float32
    assert y.shape == (100, 10)
    assert len(train_inds) == 80 and len(val_inds) == 20
    assert set(np.array(train_inds)).isdisjoint(set(np.array(val_inds)))
    # standardized per feature: near-zero mean
    assert abs(float(X.mean())) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest metanca_training/tests/metanca_training/test__fashion_mnist_prepare.py -q`
Expected: FAIL (`_prepare_fashion_mnist_arrays` not defined).

- [ ] **Step 3: Implement loader** (add near `get_mnist_datasets` in `data_utils.py`)

```python
def _prepare_fashion_mnist_arrays(images, labels, rand_key):
    """Flatten to (N,784), standard-scale, one-hot labels, 80/20 split."""
    X = np.asarray(images, dtype=np.float32).reshape(images.shape[0], -1)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    num_classes = 10
    y = np.array(nn.one_hot(np.asarray(labels, dtype=np.int32), num_classes), dtype=np.float32)

    dataset_size = X.shape[0]
    shuffled = random.permutation(rand_key, np.arange(dataset_size))
    train_size = int(0.8 * dataset_size)
    train_inds = shuffled[:train_size]
    val_inds = shuffled[train_size:]
    return X, y, train_inds, val_inds


def get_fashion_mnist_datasets(rand_key: jax.random.PRNGKey):
    """Load Fashion-MNIST via tfds and return standardized flat arrays."""
    train = tfds.as_numpy(tfds.load("fashion_mnist", split="train", batch_size=-1))
    test = tfds.as_numpy(tfds.load("fashion_mnist", split="test", batch_size=-1))
    images = np.concatenate([train["image"], test["image"]], axis=0)  # (N,28,28,1) uint8
    labels = np.concatenate([train["label"], test["label"]], axis=0)
    images = images.reshape(images.shape[0], 28, 28)
    return _prepare_fashion_mnist_arrays(images, labels, rand_key)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest metanca_training/tests/metanca_training/test__fashion_mnist_prepare.py -q`
Expected: PASS.

- [ ] **Step 5: Add the dataset config**

```yaml
# metanca_training/configs/dataset/fashion_mnist.yaml
name: fashion_mnist
input_shape: [784]
output_shape: 10
batch_size: 512
```

- [ ] **Step 6: Wire into both `load_dataset` functions**

In `metanca_training/scripts/train.py` and `metanca_training/scripts/train_regular_net.py`, add the import `get_fashion_mnist_datasets` and, in each `load_dataset`, a branch mirroring the `mnist` branch:

```python
    elif dataset_name == "fashion_mnist":
        X, y, train_inds, val_inds = get_fashion_mnist_datasets(rand_key)
        X = X.reshape(-1, *input_shape)
        train_batches, val_batches = prepare_batches(
            X, y, train_inds, val_inds, batch_size=batch_size
        )
```

(In `train_regular_net.py` use `X = _to_expected_input_layout(X, input_shape)` to match the existing pattern.)

- [ ] **Step 7: Smoke-check the loader end to end**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false python -c "import jax; from metanca_training.data_utils import get_fashion_mnist_datasets as g; X,y,tr,va=g(jax.random.key(0)); print(X.shape, y.shape, len(tr), len(va))"`
Expected: prints `(70000, 784) (70000, 10) 56000 14000`.

- [ ] **Step 8: Commit**

```bash
git add metanca_training/src/metanca_training/data_utils.py \
        metanca_training/configs/dataset/fashion_mnist.yaml \
        metanca_training/scripts/train.py metanca_training/scripts/train_regular_net.py \
        metanca_training/tests/metanca_training/test__fashion_mnist_prepare.py
git commit -m "feat: add Fashion-MNIST dataset loader and config"
```

---

### Task 4: `train_metanca` returns final params + training vars; disable early stopping

**Files:**
- Modify: `metanca_training/src/metanca_training/_train_metanca.py` (return value; early-stopping toggle).
- Modify: `metanca_training/src/metanca_training/_hydra_configs.py` (`MetaNCATrainingHyperparamsConfig`: add `early_stopping_enabled`, `early_stopping_patience`).
- Modify: `metanca_training/configs/training/default.yaml` (add the two keys).
- Test: `metanca_training/tests/metanca_training/test__train_metanca_returns.py`

**Interfaces:**
- Produces (used by Task 7): `train_metanca(...) -> tuple[chex.ArrayTree, TrainingVars]` (final `local_rule_params`, `training_vars`). Early stopping is skipped when `cfg.training.early_stopping_enabled` is `False`.

- [ ] **Step 1: Add config fields**

In `_hydra_configs.py`, inside `MetaNCATrainingHyperparamsConfig`, add:

```python
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 10
```

In `configs/training/default.yaml`, append:

```yaml
early_stopping_enabled: true
early_stopping_patience: 10
```

- [ ] **Step 2: Gate early stopping and return values in `_train_metanca.py`**

Replace the early-stopping callback construction so it is only added when enabled:

```python
    callbacks = [create_accuracy_callback()]
    if getattr(cfg.training, "early_stopping_enabled", True):
        callbacks.append(
            create_early_stopping(
                monitor="val/loss",
                patience=getattr(cfg.training, "early_stopping_patience", 10),
                mode="min",
            )
        )
    callback_runner = CallbackRunner.create(callbacks)
```

At the very end of `train_metanca` (after `callback_runner, _ = callback_runner.on_train_end(ctx)`), add:

```python
    return local_rule_params, training_vars
```

- [ ] **Step 3: Write the smoke test**

```python
# metanca_training/tests/metanca_training/test__train_metanca_returns.py
import jax
from hydra import compose, initialize_config_dir
from pathlib import Path

from metanca_training._hydra_configs import register_configs
from metanca_training import train_metanca
from metanca_training.data_utils import get_iris_datasets, prepare_batches
from metanca_training.scaling.arch_grid import build_mlp

CONFIG_DIR = str((Path(__file__).parents[3] / "configs").resolve())


def test_train_metanca_returns_params_and_vars():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=iris", "wandb=disabled",
                "training.num_metaepochs=2",
                "training.update_step_scheduler_max_steps=1",
                "training.early_stopping_enabled=false",
                "checkpoint.run_name=pytest_smoke",
                "logging.log_every_n_metaepochs=1",
            ],
        )
    key = jax.random.key(0)
    X, y, tr, va = get_iris_datasets(key)
    train_b, val_b = prepare_batches(X, y, tr, va, batch_size="all")
    models = [build_mlp((4,), n_classes=3)]  # tiny MLP on iris (4 features, 3 classes)
    params, tvars = train_metanca(
        train_b, val_b, cfg=cfg, models=models, test_model=build_mlp((4,), n_classes=3)
    )
    assert params is not None
    assert hasattr(tvars, "test_tasknet") and hasattr(tvars, "local_rule_net_apply")
```

- [ ] **Step 4: Run the test**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false pytest metanca_training/tests/metanca_training/test__train_metanca_returns.py -q`
Expected: PASS. (If iris input_shape mismatches the tiny MLP, use `dataset=iris` default `input_shape`; iris features = 4, classes = 3.)

- [ ] **Step 5: Commit**

```bash
git add metanca_training/src/metanca_training/_train_metanca.py \
        metanca_training/src/metanca_training/_hydra_configs.py \
        metanca_training/configs/training/default.yaml \
        metanca_training/tests/metanca_training/test__train_metanca_returns.py
git commit -m "feat: train_metanca returns final params + vars; toggle early stopping"
```

---

### Task 5: Per-architecture evaluator

**Files:**
- Create: `metanca_training/src/metanca_training/scaling/evaluate_pool.py`
- Test: `metanca_training/tests/metanca_training/scaling/test__evaluate_pool.py`

**Interfaces:**
- Consumes: `TrainingVars` (from Task 4), `arch_grid.build_mlp/arch_id`, `metanca.TaskNet.build`, `metanca_validation_step`, `CallbackRunner`/`create_accuracy_callback`.
- Produces (used by Task 7): `evaluate_arch_pool(*, training_vars, local_rule_params, archs, val_batches, cfg, n_update_steps=10, n_init_samples=5, rand_key) -> list[dict]` where `archs` is a list of `(widths: tuple[int,...], split: str)` and each output dict has keys `arch_id, depth, split, val_loss_mean, val_loss_std, val_acc_mean, val_acc_std`.

- [ ] **Step 1: Write the failing test** (uses untrained random rule — only checks structure/finiteness)

```python
# metanca_training/tests/metanca_training/scaling/test__evaluate_pool.py
import jax
from pathlib import Path
from hydra import compose, initialize_config_dir

from metanca_training._hydra_configs import register_configs
from metanca_training._train_metanca import before_metanca_training
from metanca_training.data_utils import get_fashion_mnist_datasets, prepare_batches
from metanca_training.scaling.arch_grid import build_mlp
from metanca_training.scaling.evaluate_pool import evaluate_arch_pool

CONFIG_DIR = str((Path(__file__).parents[4] / "configs").resolve())


def test_evaluate_arch_pool_rows():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config",
                      overrides=["dataset=fashion_mnist", "wandb=disabled",
                                 "checkpoint.run_name=pytest_eval"])
    key = jax.random.key(0)
    X, y, tr, va = get_fashion_mnist_datasets(key)
    _, val_b = prepare_batches(X, y, tr, va, batch_size=512)
    tvars = before_metanca_training(
        cfg, models=[build_mlp((32,), 10)], test_model=build_mlp((32,), 10), rand_key=key
    )
    rows = evaluate_arch_pool(
        training_vars=tvars, local_rule_params=tvars.local_rule_params,
        archs=[((32,), "train"), ((64, 32), "val")],
        val_batches=val_b, cfg=cfg, n_update_steps=1, n_init_samples=2, rand_key=key,
    )
    assert len(rows) == 2
    keys = {"arch_id", "depth", "split", "val_loss_mean", "val_loss_std",
            "val_acc_mean", "val_acc_std"}
    for r in rows:
        assert keys <= set(r)
        assert r["val_loss_mean"] == r["val_loss_mean"]  # not NaN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false pytest metanca_training/tests/metanca_training/scaling/test__evaluate_pool.py -q`
Expected: FAIL (`evaluate_pool` module not found).

- [ ] **Step 3: Implement the evaluator**

```python
# metanca_training/src/metanca_training/scaling/evaluate_pool.py
"""Evaluate a trained local rule on a pool of architectures (per-arch val metrics)."""

from typing import Sequence

import jax
import numpy as np

import metanca
from metanca_training._validation_step import metanca_validation_step
from metanca_training.callbacks import CallbackRunner, create_accuracy_callback

from .arch_grid import arch_id, build_mlp


def evaluate_arch_pool(
    *,
    training_vars,
    local_rule_params,
    archs: Sequence[tuple[Sequence[int], str]],
    val_batches,
    cfg,
    n_update_steps: int = 10,
    n_init_samples: int = 5,
    rand_key,
) -> list[dict]:
    input_shape = tuple(cfg.dataset.input_shape)
    n_spatial_dims = len(input_shape) - 1
    n_classes = int(cfg.dataset.output_shape)
    hidden_dim = training_vars.hidden_dim
    apply_fn = training_vars.local_rule_net_apply
    shared_init = (
        training_vars.test_tasknet.hidden_state_initializer,
        training_vars.test_tasknet.hidden_dim,
    )

    rows: list[dict] = []
    for widths, split in archs:
        rand_key, build_key = jax.random.split(rand_key)
        tasknet = metanca.TaskNet.build(
            model=build_mlp(widths, n_classes, use_bias=cfg.model.use_bias),
            input_shape=input_shape,
            key=build_key,
            n_spatial_dims=n_spatial_dims,
            d_neuron=cfg.positional_encoding.d_neuron,
            d_spatial=cfg.positional_encoding.d_spatial,
            d_layer=cfg.positional_encoding.d_layer,
            shared_initializer=shared_init,
        )
        losses, accs = [], []
        for s in range(n_init_samples):
            rand_key, eval_key = jax.random.split(rand_key)
            cb = CallbackRunner.create([create_accuracy_callback()])
            metrics, _ = metanca_validation_step(
                val_batches=val_batches,
                local_rule_net_apply=apply_fn,
                local_rule_params=local_rule_params,
                rand_key=eval_key,
                val_tasknet=tasknet,
                n_update_steps=n_update_steps,
                hidden_dim=hidden_dim,
                prop_cells_updated=cfg.training.prop_cells_updated,
                weight_transformer_dropout=cfg.training.weight_transformer_dropout,
                n_spatial_dims=n_spatial_dims,
                callback_runner=cb,
            )
            losses.append(float(metrics.get("loss", float("nan"))))
            accs.append(float(metrics.get("accuracy", float("nan"))))
        rows.append({
            "arch_id": arch_id(widths),
            "depth": len(widths),
            "split": split,
            "val_loss_mean": float(np.mean(losses)),
            "val_loss_std": float(np.std(losses)),
            "val_acc_mean": float(np.mean(accs)),
            "val_acc_std": float(np.std(accs)),
        })
    return rows
```

Note: `cfg.model.use_bias` exists on the composed config (defaults `True`); if absent for some compose path, replace with `True`.

- [ ] **Step 4: Run test to verify it passes**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false pytest metanca_training/tests/metanca_training/scaling/test__evaluate_pool.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add metanca_training/src/metanca_training/scaling/evaluate_pool.py \
        metanca_training/tests/metanca_training/scaling/test__evaluate_pool.py
git commit -m "feat: per-architecture pool evaluator for scaling ablation"
```

---

### Task 6: Single-run driver (`run_scaling.py`)

**Files:**
- Create: `metanca_training/scripts/scaling/__init__.py`
- Create: `metanca_training/scripts/scaling/run_scaling.py`

**Interfaces:**
- Consumes: `arch_grid.{build_grid,split_grid,sample_subset,build_mlp,N_VAL}`, `train_metanca`, `evaluate_arch_pool`, `load_dataset` (from `train.py`).
- Produces: one JSONL file appended with per-arch rows for a single `(ablation, T, rep)` run. CLI:
  `python run_scaling.py --ablation fixed5 --T 5 --rep 0 --metaepochs 12000 --out results/scaling_fixed5.jsonl [--smoke]`.
- Row schema (superset of Task 5 rows): `+ ablation, T, rep, seed`.

- [ ] **Step 1: Implement the driver**

```python
# metanca_training/scripts/scaling/run_scaling.py
"""Run one (ablation, T, rep) meta-training + per-arch evaluation for the scaling study."""

import argparse
import json
import logging
import sys
from pathlib import Path

import jax
import wandb
from hydra import compose, initialize_config_dir

# reuse the dataset loader from the training entry point
sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
from train import load_dataset  # noqa: E402

from metanca_training import train_metanca  # noqa: E402
from metanca_training._hydra_configs import register_configs  # noqa: E402
from metanca_training.scaling.arch_grid import (  # noqa: E402
    N_VAL, build_grid, build_mlp, sample_subset, split_grid,
)
from metanca_training.scaling.evaluate_pool import evaluate_arch_pool  # noqa: E402

CONFIG_DIR = str((Path(__file__).parents[2] / "configs").resolve())
logger = logging.getLogger(__name__)

SPLIT_SEED = 20260701  # fixed: the held-out V is identical across all T and reps


def build_cfg(ablation: str, run_name: str, metaepochs: int, seed: int):
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(
            config_name="config",
            overrides=[
                "dataset=fashion_mnist", "wandb=disabled",
                f"training.num_metaepochs={metaepochs}",
                "training.update_step_scheduler_type=increment",
                "training.update_step_scheduler_rate=1000",
                "training.update_step_scheduler_max_steps=10",
                "training.early_stopping_enabled=false",
                "training.sample_pooling.enabled=false",
                "checkpoint.save_top_n=1",
                f"checkpoint.run_name={run_name}",
                f"random.seed={seed}",
            ],
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", choices=["fixed5", "varying"], required=True)
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--metaepochs", type=int, default=12000)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.metaepochs = 2

    seed = 1000 * args.T + args.rep
    run_name = f"scaling_{args.ablation}_T{args.T}_rep{args.rep}"

    grid = build_grid(args.ablation)
    pool, val_archs = split_grid(grid, N_VAL[args.ablation], seed=SPLIT_SEED)
    train_archs = sample_subset(pool, args.T, seed=seed)
    logger.info("ablation=%s T=%d rep=%d | pool=%d val=%d train=%d",
                args.ablation, args.T, args.rep, len(pool), len(val_archs), len(train_archs))

    cfg = build_cfg(args.ablation, run_name, args.metaepochs, seed)
    wandb.init(mode="disabled")

    key = jax.random.key(seed)
    train_batches, val_batches = load_dataset(cfg, key)
    if args.smoke:  # keep the smoke run tiny/fast
        val_batches = tuple(v[:1] for v in val_batches)

    n_classes = int(cfg.dataset.output_shape)
    models = [build_mlp(w, n_classes) for w in train_archs]
    test_model = build_mlp(val_archs[0], n_classes)  # placeholder for in-loop monitor

    params, tvars = train_metanca(
        train_batches, val_batches, cfg=cfg, models=models, test_model=test_model
    )

    archs = [(w, "train") for w in train_archs] + [(w, "val") for w in val_archs]
    rows = evaluate_arch_pool(
        training_vars=tvars, local_rule_params=params, archs=archs,
        val_batches=val_batches, cfg=cfg, n_update_steps=10,
        n_init_samples=(1 if args.smoke else 5), rand_key=key,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for r in rows:
            f.write(json.dumps({**r, "ablation": args.ablation, "T": args.T,
                                "rep": args.rep, "seed": seed}) + "\n")
    logger.info("wrote %d rows to %s", len(rows), out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run the driver**

Run:
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python metanca_training/scripts/scaling/run_scaling.py \
  --ablation fixed5 --T 2 --rep 0 --out results/_smoke.jsonl --smoke
```
Expected: exits 0; `results/_smoke.jsonl` contains `2 (train) + 25 (val) = 27` JSON lines, each with `val_loss_mean`, `val_acc_mean`, `ablation`, `T`, `rep`, `seed`.

- [ ] **Step 3: Verify the smoke output**

Run: `python -c "import json; rows=[json.loads(l) for l in open('results/_smoke.jsonl')]; print(len(rows), sum(r['split']=='val' for r in rows))"`
Expected: prints `27 25`.

- [ ] **Step 4: Clean up smoke artifact and commit**

```bash
rm -f results/_smoke.jsonl
git add metanca_training/scripts/scaling/__init__.py metanca_training/scripts/scaling/run_scaling.py
git commit -m "feat: single-run driver for architecture-scaling ablation"
```

---

### Task 7: Memory + timing probe

**Files:**
- Create: `metanca_training/scripts/scaling/memory_probe.py`

**Interfaces:**
- Consumes: same building blocks as Task 6; runs a single meta-training run at T=100 with the pool arranged so the largest arch (`(512,512,512,512,512)`) is included, `metaepochs` small, `max_steps=10`.
- Produces: printed peak device memory and per-metaepoch wall-clock; a projected total wall-clock for the full sweep.

- [ ] **Step 1: Implement the probe**

```python
# metanca_training/scripts/scaling/memory_probe.py
"""Probe peak memory and per-metaepoch time at T=100 (largest arch included)."""

import argparse
import logging
import sys
import time
from pathlib import Path

import jax
import wandb

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
from train import load_dataset  # noqa: E402
from run_scaling import build_cfg  # noqa: E402 (same directory)

from metanca_training import train_metanca  # noqa: E402
from metanca_training.scaling.arch_grid import (  # noqa: E402
    N_VAL, build_grid, build_mlp, sample_subset, split_grid,
)

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--metaepochs", type=int, default=12, help="short run to time")
    args = p.parse_args()

    grid = build_grid("fixed5")
    pool, _ = split_grid(grid, N_VAL["fixed5"], seed=1)
    train_archs = sample_subset(pool, args.T - 1, seed=1)
    train_archs = [(512, 512, 512, 512, 512), *train_archs]  # force the largest arch in

    cfg = build_cfg("fixed5", "mem_probe", args.metaepochs, seed=0)
    wandb.init(mode="disabled")
    key = jax.random.key(0)
    train_b, val_b = load_dataset(cfg, key)
    models = [build_mlp(w, 10) for w in train_archs]

    t0 = time.time()
    train_metanca(train_b, val_b, cfg=cfg, models=models, test_model=build_mlp((32,), 10))
    dt = time.time() - t0

    per_epoch = dt / args.metaepochs
    logger.info("T=%d: %.1fs for %d metaepochs => %.2fs/metaepoch",
                args.T, dt, args.metaepochs, per_epoch)
    logger.info("Projected 12k metaepochs at T=%d: %.1f h", args.T, per_epoch * 12000 / 3600)
    for d in jax.devices():
        try:
            stats = d.memory_stats()
            logger.info("device %s peak_bytes_in_use=%.1f GB",
                        d, stats.get("peak_bytes_in_use", 0) / 1e9)
        except Exception:
            pass
```

- [ ] **Step 2: Run the probe**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false python metanca_training/scripts/scaling/memory_probe.py --T 100 --metaepochs 12`
Expected: prints per-metaepoch time, projected 12k-metaepoch hours, and peak device memory (must stay well under 120GB). **Record these numbers in `README_scaling.md`.**

- [ ] **Step 3: Commit**

```bash
git add metanca_training/scripts/scaling/memory_probe.py README_scaling.md
git commit -m "feat: memory/timing probe for T=100 scaling runs"
```

---

### Task 8: Sweep orchestrator

**Files:**
- Create: `metanca_training/scripts/scaling/run_sweep.py`

**Interfaces:**
- Produces: launches `run_scaling.py` as a **subprocess per (ablation, T, rep)** (memory isolation), sequentially. CLI:
  `python run_sweep.py --ablation fixed5 [--dry-run]` → 5 T × 3 reps = 15 subprocesses; `--out results/scaling_fixed5.jsonl`.

- [ ] **Step 1: Implement the orchestrator**

```python
# metanca_training/scripts/scaling/run_sweep.py
"""Launch the full T-sweep for one ablation as isolated subprocesses."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

TS = [1, 5, 10, 50, 100]
REPS = [0, 1, 2]
HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", choices=["fixed5", "varying"], required=True)
    p.add_argument("--metaepochs", type=int, default=12000)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out = args.out or f"results/scaling_{args.ablation}.jsonl"
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    for t in TS:
        for rep in REPS:
            cmd = [sys.executable, str(HERE / "run_scaling.py"),
                   "--ablation", args.ablation, "--T", str(t), "--rep", str(rep),
                   "--metaepochs", str(args.metaepochs), "--out", out]
            print(" ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify dry-run prints 15 commands**

Run: `python metanca_training/scripts/scaling/run_sweep.py --ablation fixed5 --dry-run | wc -l`
Expected: `15`.

- [ ] **Step 3: Commit**

```bash
git add metanca_training/scripts/scaling/run_sweep.py
git commit -m "feat: subprocess-isolated T-sweep orchestrator"
```

---

### Task 9: Adam baselines over all 246 archs

**Files:**
- Create: `metanca_training/scripts/scaling/run_adam_baselines.py`

**Interfaces:**
- Consumes: `arch_grid.{build_grid,build_mlp,arch_id}`, `data_utils.get_fashion_mnist_datasets/prepare_batches`, `regular_net_functions.{init_and_train_regular_network_batched,evaluate_reg_net_batched}`, `_loss.compute_loss`.
- Produces: `results/adam_baselines_fashion_mnist.json` — `{arch_id: {"val_loss", "val_acc", "train_acc", "depth"}}`. CLI: `python run_adam_baselines.py --epochs 50 [--smoke]`.

- [ ] **Step 1: Implement the baseline runner**

```python
# metanca_training/scripts/scaling/run_adam_baselines.py
"""Train each grid architecture with Adam; save a val-performance table."""

import argparse
import json
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from metanca_training.data_utils import get_fashion_mnist_datasets, prepare_batches
from metanca_training.regular_net_functions import (
    _unpack_model_state, evaluate_reg_net_batched, init_and_train_regular_network_batched,
)
from metanca_training._loss import compute_loss
from metanca_training.scaling.arch_grid import arch_id, build_grid, build_mlp

logger = logging.getLogger(__name__)


def _val_loss(model, model_state, val_batches) -> float:
    params, _ = _unpack_model_state(model_state)   # no-bn MLP -> raw params pytree
    apply_fn = lambda p, xx: model.apply({"params": p}, xx)
    total, n = 0.0, val_batches[0].shape[0]
    for i in range(n):
        x = val_batches[0][i].astype(jnp.float32)
        total += float(compute_loss(x, val_batches[1][i], val_batches[2][i], params, apply_fn))
    return total / max(n, 1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--out", type=str, default="results/adam_baselines_fashion_mnist.json")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    key = jax.random.key(0)
    X, y, tr, va = get_fashion_mnist_datasets(key)
    train_b, val_b = prepare_batches(X, y, tr, va, batch_size=512)

    # union of both grids == varying grid (246 archs); fixed5 ⊂ varying
    grid = build_grid("varying")
    if args.smoke:
        grid = grid[:3]
        args.epochs = 1

    results: dict[str, dict] = {}
    for i, widths in enumerate(grid):
        key, mk = jax.random.split(key)
        model = build_mlp(widths, 10)
        _, model_state, _, model, _, _ = init_and_train_regular_network_batched(
            mk, None, train_b, val_b, epochs=args.epochs, model=model,
            optimizer=optax.adam(1e-3),
        )
        val_acc = float(evaluate_reg_net_batched(model, model_state, val_b))
        train_acc = float(evaluate_reg_net_batched(model, model_state, train_b))
        val_loss = _val_loss(model, model_state, val_b)
        results[arch_id(widths)] = {
            "val_loss": val_loss, "val_acc": val_acc,
            "train_acc": train_acc, "depth": len(widths),
        }
        logger.info("[%d/%d] %s val_acc=%.4f val_loss=%.4f",
                    i + 1, len(grid), arch_id(widths), val_acc, val_loss)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    logger.info("wrote %d arch baselines to %s", len(results), out)


if __name__ == "__main__":
    main()
```

Note: `init_and_train_regular_network_batched` returns `model_state` already in the form `evaluate_reg_net_batched` and `compute_loss` accept (params pytree for a no-bn MLP). If `_val_loss` raises on the model-state form, unwrap via `metanca_training.regular_net_functions._unpack_model_state(model_state)[0]`.

- [ ] **Step 2: Smoke-run**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false python metanca_training/scripts/scaling/run_adam_baselines.py --smoke --out results/_adam_smoke.json`
Expected: exits 0; JSON has 3 entries each with `val_loss`, `val_acc`, `train_acc`, `depth`.

- [ ] **Step 3: Clean up and commit**

```bash
rm -f results/_adam_smoke.json
git add metanca_training/scripts/scaling/run_adam_baselines.py
git commit -m "feat: Adam baseline table over the full architecture grid"
```

---

### Task 10: Plotting

**Files:**
- Create: `metanca_training/scripts/scaling/plot_scaling.py`
- Test: `metanca_training/tests/metanca_training/scaling/test__plot_scaling.py`

**Interfaces:**
- Produces:
  - `load_results(path: str) -> list[dict]` — parse a scaling JSONL.
  - `aggregate(rows: list[dict], metric: str) -> dict[int, dict[str, list[float]]]` — `metric` ∈ `{"loss","acc"}`; returns `{T: {"train": [...], "val": [...]}}` pooling all reps' per-arch means.
  - `plot_boxplots(agg, metric, out_path)` and `plot_scatter(agg, metric, out_path, adam=None)` — write PNGs (two-panel shared-Y boxplot; central-tendency scatter). CLI writes both metrics' figures for one ablation.

- [ ] **Step 1: Write the failing test** (pure `load_results`/`aggregate` on a fixture; plotting just asserts the file is created)

```python
# metanca_training/tests/metanca_training/scaling/test__plot_scaling.py
import json
from pathlib import Path

from metanca_training.scaling.plotting import aggregate, load_results


def _write(tmp_path):
    rows = [
        {"T": 1, "rep": 0, "split": "train", "val_loss_mean": 0.5, "val_acc_mean": 0.8},
        {"T": 1, "rep": 0, "split": "val", "val_loss_mean": 0.9, "val_acc_mean": 0.6},
        {"T": 5, "rep": 0, "split": "train", "val_loss_mean": 0.4, "val_acc_mean": 0.85},
        {"T": 5, "rep": 0, "split": "val", "val_loss_mean": 0.7, "val_acc_mean": 0.7},
    ]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_load_and_aggregate(tmp_path):
    rows = load_results(_write(tmp_path))
    agg = aggregate(rows, "loss")
    assert set(agg) == {1, 5}
    assert agg[1]["val"] == [0.9]
    assert agg[5]["train"] == [0.4]


def test_plot_writes_png(tmp_path):
    from metanca_training.scaling.plotting import plot_boxplots
    agg = aggregate(load_results(_write(tmp_path)), "acc")
    out = tmp_path / "box.png"
    plot_boxplots(agg, "acc", out)
    assert out.exists() and out.stat().st_size > 0
```

Note: the testable logic lives in the importable package module `metanca_training/src/metanca_training/scaling/plotting.py` (Step 3); the `scripts/scaling/plot_scaling.py` CLI (Step 4) is a thin wrapper that imports from it. This keeps plotting logic unit-testable without invoking the CLI.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest metanca_training/tests/metanca_training/scaling/test__plot_scaling.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement plotting logic** in `metanca_training/src/metanca_training/scaling/plotting.py`

```python
# metanca_training/src/metanca_training/scaling/plotting.py
"""Load scaling results and render boxplot + scatter figures per metric."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_MEAN_KEY = {"loss": "val_loss_mean", "acc": "val_acc_mean"}
_LABEL = {"loss": "validation loss", "acc": "validation accuracy"}


def load_results(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def aggregate(rows: list[dict], metric: str) -> dict[int, dict[str, list[float]]]:
    key = _MEAN_KEY[metric]
    agg: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        agg.setdefault(int(r["T"]), {"train": [], "val": []})[r["split"]].append(float(r[key]))
    return dict(sorted(agg.items()))


def plot_boxplots(agg, metric, out_path, adam: dict | None = None) -> None:
    ts = list(agg)
    fig, (ax_tr, ax_va) = plt.subplots(1, 2, sharey=True, figsize=(11, 4))
    for ax, split, title in ((ax_tr, "train", "train archs"), (ax_va, "val", "val archs")):
        ax.boxplot([agg[t][split] for t in ts], tick_labels=[str(t) for t in ts])
        ax.set_title(f"{title} — {_LABEL[metric]}")
        ax.set_xlabel("T (number of training architectures)")
        if adam is not None:
            vals = [adam[k][ "val_loss" if metric == "loss" else "val_acc"] for k in adam]
            ax.axhline(float(np.median(vals)), ls="--", color="tab:red",
                       label="Adam median")
            ax.legend()
    ax_tr.set_ylabel(_LABEL[metric])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(agg, metric, out_path, adam: dict | None = None) -> None:
    ts = list(agg)
    xs = [float(np.mean(agg[t]["train"])) for t in ts]
    ys = [float(np.mean(agg[t]["val"])) for t in ts]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(xs, ys, "-o")
    for t, x, y in zip(ts, xs, ys):
        ax.annotate(f"T={t}", (x, y), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel(f"mean train-arch {_LABEL[metric]}")
    ax.set_ylabel(f"mean val-arch {_LABEL[metric]}")
    ax.set_title(f"Architecture scaling — {_LABEL[metric]}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 4: Add the CLI wrapper** `metanca_training/scripts/scaling/plot_scaling.py`

```python
# metanca_training/scripts/scaling/plot_scaling.py
"""CLI: render loss+accuracy boxplot & scatter figures for one ablation."""

import argparse
import json
from pathlib import Path

from metanca_training.scaling.plotting import (
    aggregate, load_results, plot_boxplots, plot_scatter,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--adam", default=None)
    p.add_argument("--outdir", default="results/figures")
    p.add_argument("--tag", default="fixed5")
    args = p.parse_args()

    rows = load_results(args.results)
    adam = json.loads(Path(args.adam).read_text()) if args.adam else None
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for metric in ("loss", "acc"):
        agg = aggregate(rows, metric)
        plot_boxplots(agg, metric, outdir / f"{args.tag}_box_{metric}.png", adam)
        plot_scatter(agg, metric, outdir / f"{args.tag}_scatter_{metric}.png", adam)
    print(f"wrote figures to {outdir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest metanca_training/tests/metanca_training/scaling/test__plot_scaling.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add metanca_training/src/metanca_training/scaling/plotting.py \
        metanca_training/scripts/scaling/plot_scaling.py \
        metanca_training/tests/metanca_training/scaling/test__plot_scaling.py
git commit -m "feat: scaling-study plotting (boxplots + scatter, per metric)"
```

---

### Task 11: Full test pass + run documentation

**Files:**
- Modify: `README_scaling.md` (final run commands, in order).

- [ ] **Step 1: Run the whole suite**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false pytest metanca/ metanca_training/ -q`
Expected: PASS.

- [ ] **Step 2: Document the run order in `README_scaling.md`**

Record the exact commands (probe → Adam baselines → both sweeps → plots), e.g.:

```bash
# 3. probe (record numbers)
python metanca_training/scripts/scaling/memory_probe.py --T 100 --metaepochs 12
# 4. Adam baselines
python metanca_training/scripts/scaling/run_adam_baselines.py --epochs 50
# 5. fixed-depth sweep
python metanca_training/scripts/scaling/run_sweep.py --ablation fixed5
# 6. varying-depth sweep
python metanca_training/scripts/scaling/run_sweep.py --ablation varying
# 7. plots
python metanca_training/scripts/scaling/plot_scaling.py \
  --results results/scaling_fixed5.jsonl \
  --adam results/adam_baselines_fashion_mnist.json --tag fixed5
python metanca_training/scripts/scaling/plot_scaling.py \
  --results results/scaling_varying.jsonl \
  --adam results/adam_baselines_fashion_mnist.json --tag varying
```

- [ ] **Step 3: Commit**

```bash
git add README_scaling.md
git commit -m "docs: architecture-scaling study run order and recorded probe numbers"
```

---

## Notes for the executor

- **Always** export `XLA_PYTHON_CLIENT_PREALLOCATE=false` before any JAX process (unified memory on GB10).
- Tasks 2, 3, 5, 10 have real unit tests (TDD). Tasks 1, 6, 7, 8, 9 are script/integration tasks verified by smoke runs — treat a clean `--smoke`/`--dry-run` exit + expected output as the gate.
- The `results/` directory holds JSONL + JSON + figures; it is a data output dir. Add it to `.gitignore` if the team does not want large result files committed (confirm before committing real run outputs).
- If the GB10 probe (Task 7) shows T=100 wall-clock is impractical, **do not** silently trim — surface the projected time to the user (the spec forbids trimming reps without approval).
```
