# Architecture-Scaling Ablation for MetaNCA — Design

Status: approved (2026-07-01)
Branch: `architecture-scaling-ablation`

## Goal

Characterize how MetaNCA's architecture generalization scales with the **number of
training architectures T**. For a fixed dataset (Fashion-MNIST) and a fixed grid of
architecture specs, we train one MetaNCA local rule on T architectures and measure the
per-architecture validation-set performance on both the T training architectures and a
fixed held-out set of V validation architectures. Sweeping T reveals the statistics of
the performance distribution (central tendencies + full distributions via boxplots).

Two ablations:
1. **Fixed-depth (D=5)** — primary. Isolates width generalization at constant depth.
2. **Varying-depth (D=2..5)** — mixed-depth generalization.

## Architecture family (shared)

- Hidden layer widths are drawn from powers of two: **{32, 64, 128, 256, 512}**.
- Hidden widths are **non-increasing** toward the output ("pyramidal"); constant-width
  is allowed as the degenerate case (e.g. `[512,512,512,512,512]`).
- Output layer = number of classes (**10** for Fashion-MNIST), appended after the hidden
  widths.
- Instantiated as
  `MultiLayerPerceptron(layer_specs=[*widths, 10], activation="leaky_relu",
  final_layer_activation="identity", use_bias=true)` — matching repo conventions
  (`configs/model/mnist.yaml`).
- Each architecture has a deterministic string ID derived from its widths, e.g.
  depth-3 arch with widths `(512,128,64)` → `d3_512-128-64`.

### Counts (combinations-with-repetition, `C(L+4, L)`)

| Depth L | Count |
|---|---|
| 2 | 15 |
| 3 | 35 |
| 4 | 70 |
| 5 | 126 |

- Fixed-depth grid (D=5): **126**.
- Varying-depth grid (D=2..5): 15+35+70+126 = **246**.

## Ablations and splits

The V validation architectures are a random sample from the grid, **held fixed** across
all values of T and all replicates within an ablation.

### A. Fixed-depth (D=5) — first / primary
- Grid = all 126 depth-5 archs.
- Hold out **V = 25** (~20%) at random as the fixed validation set.
- Training pool = **101** archs.
- Sweep **T ∈ {1, 5, 10, 50, 100}** (T=100 ≈ 80% of the grid).

### B. Varying-depth (D=2..5) — second
- Grid = all 246 archs.
- Hold out **V = 49** (~20%) at random as the fixed validation set.
- Training pool = **197** archs.
- Sweep **T ∈ {1, 5, 10, 50, 100}** (a full-pool T may be added later; out of scope now).

### Replicates and sampling
- For each T: **3 independent replicate runs**. Each replicate draws a fresh, independent
  random T-subset from the training pool (distinct seed per replicate).
- Total meta-training runs = 5 T × 3 reps × 2 ablations = **30**, plus one Adam baseline
  pass over all 246 archs.
- **No trimming of replicates** even if T=100 is expensive (explicit decision). The
  memory/timing probe still runs first to project wall-clock.

## Training protocol (per run)

- Dataset: Fashion-MNIST, flattened to `input_shape=[784]`, `output_shape=10`,
  `batch_size=512`, 80/20 train/val split of the data (standard-scaled, one-hot labels).
- One MetaNCA local rule meta-trained on the T architectures (the "pool"), using the
  existing **single-GPU sequential accumulation** path (`_accumulate_metanca_gradients`)
  — the only path on this single-GPU GB10.
- **12,000 metaepochs.** Update-step schedule: `increment`, **rate=1000**, **max_steps=10**
  → the number of BPTT steps reaches 10 at metaepoch ~9k–10k; the remaining ~2–3k
  metaepochs let the 10-step regime converge.
- **Early stopping disabled** (fixed budget): set the early-stopping patience ≥
  `num_metaepochs` (or add a disable flag).
- Sample pooling stays **off** (it multiplies resident tasknet state by `pool_size`,
  which we cannot afford at T=100 on 120GB).
- Checkpoints saved via the existing manager (ranked by the existing train-data monitor;
  no validation-arch leakage into checkpoint selection).

## Per-architecture evaluation (after training)

Decoupled from the training loop (mirrors `eval_arch_generalization` in `train.py`):

- Load the **final checkpoint** (step 12k) of the run. (Post-hoc selection by a training
  metric is possible later but the default is the final checkpoint.)
- For each architecture (all T train archs **and** all V val archs):
  - Rebuild its TaskNet using the **shared hidden-state initializer** from the trained
    run (as `eval_arch_generalization` does), so evaluation matches training-time
    construction.
  - Roll out the local rule for **10** update steps (the saturated value).
  - Compute **val loss** and **val accuracy** on the data validation set, averaged over
    **5 random inits** per arch (record mean and std).
- Write one row per arch to `results/scaling_<ablation>.jsonl` with fields:
  `ablation, T, rep, seed, arch_id, depth, split ∈ {train, val}, val_loss_mean,
  val_loss_std, val_acc_mean, val_acc_std`.

## Adam baselines

- A standalone runner trains each of the **246** archs with Adam (lr 1e-3, fixed epoch
  budget — target a few minutes total; tune epochs so archs reach reasonable accuracy)
  and records **val loss + val accuracy** per arch.
- Saved as a **table** `results/adam_baselines_fashion_mnist.json`
  (`arch_id → {val_loss, val_acc, train_acc, depth}`), independent of the MetaNCA runs.
- Imported at plotting time; optionally overlaid as a reference band.

## Plots

Generated **per metric on separate figures** — validation **loss** and validation
**accuracy** each get their own full plot set (the two metrics never share a plot):

1. **Two-panel boxplot** (shared Y axis): left panel = distribution of the train-arch
   val-metric per T; right panel = distribution of the val-arch val-metric per T. One box
   per T in each panel.
2. **Central-tendency scatter**: X = mean (and median) train-arch val-metric, Y = mean
   (and median) val-arch val-metric, one point per T.
3. Optional Adam reference band (median + IQR across archs) overlaid on the above.

Plots read from `results/scaling_<ablation>.jsonl` and (optionally)
`results/adam_baselines_fashion_mnist.json`.

## Code layout

Pure/testable logic under `src/`, executable entry points under `scripts/scaling/`:

- `src/metanca_training/scaling/arch_grid.py` — enumerate archs (non-increasing
  power-of-two widths), build a grid, split train/val (seeded), sample T-subsets
  (seeded), arch → `layer_specs`, arch → ID. **Unit-tested**: grid counts (126, 246),
  non-increasing invariant, determinism of split/sample.
- `src/metanca_training/scaling/evaluate_pool.py` — per-arch evaluator (loads checkpoint,
  builds tasknets with shared initializer, rolls out 10 steps, averages over inits).
- `data_utils.get_fashion_mnist_datasets(...)` — Fashion-MNIST loader (flattened,
  standard-scaled, one-hot, 80/20 split), plus `configs/dataset/fashion_mnist.yaml`,
  wired into `load_dataset` in both `train.py` and `train_regular_net.py`.
- `scripts/scaling/memory_probe.py` — run a single metaepoch at T=100 with the pool that
  includes the largest arch; log peak memory + per-metaepoch time; project total runtime.
- `scripts/scaling/run_adam_baselines.py` — Adam over all 246 archs → JSON table.
- `scripts/scaling/run_scaling.py` — driver: for each (ablation, T, rep) build the train
  arch list, run meta-training (12k), then run `evaluate_pool` on train+val archs;
  append to `results/scaling_<ablation>.jsonl`.
- `scripts/scaling/plot_scaling.py` — produce the boxplot + scatter figures per metric.

## Environment (uv)

- Set up with `uv venv` + `uv pip install` (jax CUDA, tensorflow-cpu, and the two
  editable packages `metanca`, `metanca_training`).
- Verify: `python -c "import jax; print(jax.devices())"` sees the GB10 GPU, and
  `pytest metanca/ metanca_training/` passes.

**Risk (flagged):** the host is aarch64 (Grace CPU) + Blackwell (GB10) with 120GB
**unified** memory (CPU+GPU share one pool). Standard `jax[cuda12]` pip wheels may not
support this aarch64/Blackwell combination; NVIDIA's JAX build/container may be required.
Resolve empirically in the environment step **before** any other work. Because memory is
unified, JAX GPU preallocation should likely be disabled/limited
(`XLA_PYTHON_CLIENT_PREALLOCATE=false` or a bounded `XLA_PYTHON_CLIENT_MEM_FRACTION`) so
the GPU allocator does not starve host RAM.

## Feasibility / memory

- Single-GPU path processes archs **sequentially** with `@jax.checkpoint` gradient
  checkpointing on the BPTT scan → peak activation memory scales with the **largest single
  arch**, not with T.
- What scales with T is **resident tasknet state** (params + hidden states + positional
  encodings for all T archs). For flat Fashion-MNIST, `hidden_dim = 2·d_neuron + d_layer
  = 30`. The largest depth-5 arch (`[512×5,10]`) ≈ 1.46M params ≈ ~350MB resident;
  even 100 such nets ≈ ~35GB, and most archs are far smaller. Feasible within 120GB with
  sample pooling off. The binding cost is **wall-clock** (per-metaepoch loop is O(T)),
  which the probe quantifies.

## Order of operations

1. uv environment + verify JAX sees the GB10 + `pytest` passes.
2. Fashion-MNIST loader + `dataset/fashion_mnist.yaml` wired in.
3. **Memory + timing probe** at T=100 (project total wall-clock; do not trim reps).
4. Adam baselines over all 246 archs → JSON table.
5. Fixed-depth ablation (15 runs + per-arch eval) → `results/scaling_fixed5.jsonl`.
6. Varying-depth ablation (15 runs + per-arch eval) → `results/scaling_varying.jsonl`.
7. Plots (loss and accuracy figure sets) for both ablations.

## Out of scope (YAGNI)

- Datasets other than Fashion-MNIST (MNIST/CIFAR-10 mentioned but deferred).
- Multi-GPU / distributed execution.
- Full-pool T points beyond {1,5,10,50,100} (varying-depth full-pool T deferred).
- Architecture families beyond non-increasing power-of-two-width MLPs (conv/ResNet).
