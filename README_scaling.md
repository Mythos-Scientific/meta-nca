# Architecture-Scaling Ablation — environment & run notes

Study docs:
- Spec: `docs/superpowers/specs/2026-07-01-architecture-scaling-ablation-design.md`
- Plan: `docs/superpowers/plans/2026-07-01-architecture-scaling-ablation.md`

## Host

- NVIDIA **GB10** (Grace-Blackwell), single GPU, **aarch64**, ~120 GB **unified** memory
  (CPU + GPU share one pool).
- Always export `XLA_PYTHON_CLIENT_PREALLOCATE=false` before any JAX process (unified memory).

## Environment (uv)

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install --python .venv/bin/python \
    -r requirements.in -r metanca/requirements.in -r metanca_training/requirements.in
VIRTUAL_ENV=.venv uv pip install --python .venv/bin/python -e ./metanca -e ./metanca_training
VIRTUAL_ENV=.venv uv pip install --python .venv/bin/python -r requirements-dev.in   # pytest, etc.
```

Run everything through the venv interpreter explicitly (shell state does not persist between
this repo's tooling calls): `/home/dan/Projects/meta-nca/.venv/bin/python`.

## Findings that shaped the study

- **JAX GPU works with stock wheels.** `jax[cuda12]` (jax/jaxlib **0.8.0**, nvidia-cuda-*-cu12
  12.9) installs on aarch64 and sees the GB10:
  `XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -c "import jax; print(jax.devices())"`
  → `[CudaDevice(id=0)]`, and a matmul runs. No NVIDIA-specific JAX build was needed.
- **TensorFlow is NOT installable on aarch64** here: `tensorflow-cpu` has no aarch64/py3.12
  wheel, and `tfds.load(...)` requires TensorFlow. Therefore **Fashion-MNIST is loaded via
  scikit-learn OpenML** (`fetch_openml("Fashion-MNIST", version=1, ...)` → 70000×784 uint8-range
  floats, 10 classes), mirroring the existing `fetch_openml("mnist_784")` MNIST loader — no TF
  dependency. (`import tensorflow_datasets` still succeeds lazily, so unrelated imports are fine.)

## Running tests

The pre-existing suites import a top-level `tests` package, so they must be run **from within
each package directory** (each package's `pyproject.toml` sets `pythonpath = ["."]`):

```bash
(cd metanca          && XLA_PYTHON_CLIENT_PREALLOCATE=false ../.venv/bin/python -m pytest -q)
(cd metanca_training && XLA_PYTHON_CLIENT_PREALLOCATE=false ../.venv/bin/python -m pytest -q)
```

Baseline at env setup: metanca 76 passed / 4 skipped; metanca_training 44 passed.

New scaling tests live under `metanca_training/tests/metanca_training/scaling/`; run them the same
way from the `metanca_training/` directory, e.g.:

```bash
cd metanca_training && XLA_PYTHON_CLIENT_PREALLOCATE=false \
    ../.venv/bin/python -m pytest tests/metanca_training/scaling/ -q
```

## Run order (filled in as the study is built)

1. Environment (this file). ✅
2. Fashion-MNIST loader + config.
3. Memory + timing probe at T=100 — record per-metaepoch time, projected 12k-metaepoch hours,
   peak device memory here:
   - _TBD after Task 7_
4. Adam baselines over all 246 archs.
5. Fixed-depth (D=5) sweep.
6. Varying-depth (D=2..5) sweep.
7. Plots (loss + accuracy) for both ablations.
