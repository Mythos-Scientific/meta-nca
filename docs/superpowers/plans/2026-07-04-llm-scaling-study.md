# LLM Architecture-Scaling Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the monorepo's weight-level `TinyCausalLM` MetaNCA support into the public repo and run the T-scaling study on Shakespeare over a 68-arch `(d_model, heads, vocab)` grid, measuring val_loss / val_perplexity / val_bpb.

**Architecture:** Copy the model + elementwise message kernels + metrics/loaders from `/home/dan/Projects/monorepo`; carefully merge 4 core-library files (multi-role params, int32 dummy inputs, constvar matching, fwd/bwd slice strategies) while preserving the public `shared_initializer`; add a multi-tokenizer collate data pipeline, an LLM grid module, per-arch batch streams in the train loop, an LLM evaluator/runner mirroring the MLP study's durable-run contract; probe before launching.

**Tech Stack:** Python 3.12, JAX 0.8/Flax 0.12 (existing env pins), Hydra, SentencePiece (new dep), orbax, wandb, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-04-llm-scaling-study-design.md`. Branch: `llm-scaling`, worktree `/home/dan/Projects/meta-nca/.worktrees/llm-scaling` (NEVER touch the main checkout at `/home/dan/Projects/meta-nca` — the MLP study runs from it).
- Worktree venv: `/home/dan/Projects/meta-nca/.worktrees/llm-scaling/.venv` (Task 1 creates it). All commands use it, absolute path.
- **All tests run CPU-only:** prefix every pytest/python with `JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false`. GPUs belong to the running MLP study; only the Task 12 probe uses a GPU (controller-run).
- Tests run from within each package dir (`metanca/` or `metanca_training/`), e.g. `cd metanca_training && JAX_PLATFORMS=cpu ../.venv/bin/python -m pytest tests/... -q`.
- Regression gate on every core-library task: `metanca` suite stays 76 passed/4 skipped(+new), `metanca_training` scaling tests stay green.
- Monorepo source root: `/home/dan/Projects/monorepo` (read-only; never modify it).
- Grid constants (verbatim): `D_MODELS=(32,48,64,96,128,192,256)`, `HEADS=(2,4,8)`, head_dim=`d_model/heads` integer in `[8,64]`, `VOCABS=(512,1024,2048,4096)` → 17 combos × 4 = 68 archs; `N_VAL_LLM=14`; depth `num_layers=3`; `mlp_dim=4*d_model`; `context_length=64`; `batch_size=8`; arch id `llm_d{d}_h{h}_v{v}`.
- `TaskNet.build_many` MUST keep its `shared_initializer` parameter and default behavior (the monorepo removed it; do not carry that removal over).
- bpb definition (verbatim): `bpb = mean_token_CE_nats × (N_scored_tokens / N_bytes) / ln(2)`, factors computed per vocab per split.
- Run contract for the runner: per-run JSONL under `results/scaling/llm/T{T}_rep{rep}.jsonl`, one flushed row per arch, `.done` marker after eval, resume via skip_ids + orbax checkpoint, wandb run `scaling_llm_T{T}_rep{rep}` with `config.study="llm_shakespeare"`.
- Commit after every task with the message given in the task.

---

### Task 1: Worktree environment

**Files:**
- Create: `.venv/` in the worktree (git-ignored)

**Interfaces:**
- Produces: `/home/dan/Projects/meta-nca/.worktrees/llm-scaling/.venv/bin/python` with `metanca` + `metanca_training` installed editable FROM THE WORKTREE, plus dev deps and `sentencepiece`.

- [ ] **Step 1: Create venv + install**

```bash
cd /home/dan/Projects/meta-nca/.worktrees/llm-scaling
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install --python .venv/bin/python \
  -r requirements.in -r metanca/requirements.in -r metanca_training/requirements.in
VIRTUAL_ENV=$PWD/.venv uv pip install --python .venv/bin/python -e ./metanca -e ./metanca_training
VIRTUAL_ENV=$PWD/.venv uv pip install --python .venv/bin/python -r requirements-dev.in sentencepiece
```

- [ ] **Step 2: Verify imports resolve to the worktree (not the main checkout)**

Run: `.venv/bin/python -c "import metanca, metanca_training; print(metanca.__file__)"`
Expected: path starts with `/home/dan/Projects/meta-nca/.worktrees/llm-scaling/`.

- [ ] **Step 3: Baseline test pass (CPU)**

```bash
(cd metanca && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false ../.venv/bin/python -m pytest -q)
(cd metanca_training && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false ../.venv/bin/python -m pytest tests/metanca_training/scaling/test__arch_grid.py -q)
```
Expected: metanca 76 passed/4 skipped (CPU may skip a few more — record the CPU baseline count in your report; it is the reference for later regression checks); arch_grid 9 passed.

- [ ] **Step 4: Add sentencepiece to requirements and commit**

Append `sentencepiece` on its own line to `metanca_training/requirements.in`.

```bash
git add metanca_training/requirements.in
git commit -m "chore: worktree env for llm-scaling; add sentencepiece dep"
```

---

### Task 2: Copy TinyCausalLM + elementwise message kernels

**Files:**
- Create: `metanca/src/metanca/nn/_tiny_causal_lm.py` (copy of `/home/dan/Projects/monorepo/metanca/src/metanca/nn/_tiny_causal_lm.py`, verbatim)
- Create: `metanca/tests/metanca/nn/test__tiny_causal_lm.py` (copy of `/home/dan/Projects/monorepo/metanca/tests/metanca/nn/test__tiny_causal_lm.py`)
- Modify: `metanca/src/metanca/neighbors/_message_kernels/_elementwise_nonbarrier_op.py`, `_add.py`, `_mul.py`, `_sub.py` — replace each with the monorepo version at `/home/dan/Projects/monorepo/metanca/src/metanca/neighbors/_message_kernels/<same name>` (these add `_classify_elementwise` with the `shared_trailing_channel` case and `register_backward_slice_strategy` variants)
- Modify: `metanca/src/metanca/nn/__init__.py` and `metanca/src/metanca/__init__.py` — add `TinyCausalLM` export exactly as the monorepo's exports do (do NOT add `NeuronTaskNet`/`NeuronRuleNet`/`WeightRuleNet`/`update_tasknet_neuron_level`)

**Interfaces:**
- Produces: `metanca.nn.TinyCausalLM(vocab_size, d_model, num_heads, num_layers, mlp_dim, max_seq_len)`; upgraded elementwise kernels with backward strategies.

- [ ] **Step 1: Copy the files listed above** (diff each kernel file against monorepo after copying to confirm byte-identical).
- [ ] **Step 2: Run the copied model test**

Run: `cd metanca && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false ../.venv/bin/python -m pytest tests/metanca/nn/test__tiny_causal_lm.py -q`
Expected: PASS (all tests in that file).

- [ ] **Step 3: Regression — full metanca suite**

Run: `cd metanca && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false ../.venv/bin/python -m pytest -q`
Expected: baseline count from Task 1 + the new TinyCausalLM tests, 0 failures. NOTE: if the kernel replacements break an existing message-kernel test, STOP and report BLOCKED with the diff — do not adapt tests to make them pass.

- [ ] **Step 4: Commit**

```bash
git add metanca/src/metanca metanca/tests
git commit -m "feat: port TinyCausalLM + upgraded elementwise message kernels from monorepo"
```

---

### Task 3: Core merge — neighbors, initializer, get_neighbors (multi-role support)

**Files:**
- Modify: `metanca/src/metanca/neighbors/_convert_parameter_graph.py` — merge from `/home/dan/Projects/monorepo/metanca/src/metanca/neighbors/_convert_parameter_graph.py`: add the `slice_strategy` parameter (default `apply_forward_slice_strategy`) with edge validation (drop edges raising `TypeError/ValueError/KeyError/IndexError` with a debug log), the `if var in var2data` constvar guard, and `convert_parameter_graph_backward` (same fn with `apply_backward_slice_strategy`). Export it from `metanca/src/metanca/neighbors/__init__.py` the same way the monorepo does.
- Modify: `metanca/src/metanca/hidden_state/_initializer_from_parameters.py` — merge from monorepo: `get_layer_names_from_shapes` regex → `\.(?:bias|kernel|embedding|scale)$`; `create_unified_initializer` + `initializer_from_parameters` pick primary shapes from kernel/embedding/scale (fall back to any non-bias key) instead of assuming `.kernel`.
- Modify: `metanca/src/metanca/nn/_get_neighbors.py` — merge from monorepo: widen `param_type` to `Literal["bias","kernel","embedding","scale"]`, add `("embedding","fwd"/"bwd")` and `("scale", _)` cases, explicit `raise ValueError` fallthrough.
- Test: `metanca/tests/metanca/hidden_state/test__multi_role_layer_names.py` (new)

**Interfaces:**
- Consumes: monorepo files above (read them side-by-side with the public versions; apply the monorepo's deltas without dropping public-only code).
- Produces: `convert_parameter_graph_backward(...)`; multi-role layer-name/shape handling used by Task 4.

- [ ] **Step 1: Write the failing test**

```python
# metanca/tests/metanca/hidden_state/test__multi_role_layer_names.py
from metanca.hidden_state import get_layer_names_from_shapes


def test_layer_names_include_embedding_and_scale():
    shapes = {
        "token_embed.embedding": (512, 32),
        "blocks_0.attn_norm.scale": (32,),
        "blocks_0.self_attn_qkv.kernel": (32, 96),
        "final_norm.scale": (32,),
        "lm_head.kernel": (32, 512),
    }
    names = get_layer_names_from_shapes(shapes)
    assert names == ["token_embed", "blocks_0.attn_norm", "blocks_0.self_attn_qkv",
                     "final_norm", "lm_head"]
```

(If `get_layer_names_from_shapes` isn't exported from `metanca.hidden_state`, import from the private module path the public repo uses today.)

- [ ] **Step 2: Run it to confirm it FAILS on the current code** (embedding/scale names will be mangled by the kernel/bias-only regex).
- [ ] **Step 3: Apply the three merges.** For each file: open the monorepo version and the public version; port the monorepo's changes onto the public file. The public files have no local divergence in these three modules EXCEPT check `hidden_state/__init__.py` exports — keep public exports intact and add any new ones the monorepo has.
- [ ] **Step 4: Test passes + regression**

```bash
cd metanca && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false ../.venv/bin/python -m pytest tests/metanca/hidden_state/test__multi_role_layer_names.py -q
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false ../.venv/bin/python -m pytest -q
```
Expected: new test PASS; full suite = Task 2 count + 1, 0 failures.

- [ ] **Step 5: Commit**

```bash
git add metanca/src metanca/tests
git commit -m "feat: multi-role param support in neighbors/initializer/get_neighbors (+backward slice strategies)"
```

---

### Task 4: Core merge — `_tasknet.py` (the critical one)

**Files:**
- Modify: `metanca/src/metanca/nn/_tasknet.py` — merge from `/home/dan/Projects/monorepo/metanca/src/metanca/nn/_tasknet.py`. Apply exactly these monorepo deltas, and NOTHING else:
  1. `dummy_input_dtype` field on the `TaskNet` dataclass + parameter on `build` and `build_many` (default `jnp.float32`), used in `jnp.zeros((1, *input_shape), dtype=...)` probes in `build` and `reset`.
  2. Multi-role hidden-state/PE init in `build` and `reset`: `primary_param_priority = ("kernel", "embedding", "scale", "bias")`; EVERY param role present per layer (kernel/bias/embedding/scale) gets a hidden state + positional encoding.
  3. Identity-based constvar↔param matching in `build` (match `jaxpr` constvars to params via `id()` of arrays against `closed.literals`; drop non-parameter constvars) replacing the order-based `zip`.
  4. Forward/backward graph split: forward adjacency via `convert_parameter_graph`, backward via `convert_parameter_graph_backward` (from Task 3).
  5. `iter_param_names()` yields `("bias", "kernel", "embedding", "scale")`.
  6. `_fix_conv_dense_boundary_pe` skips layers without `.kernel`.
  **PRESERVE (public-only, monorepo lacks them): the `shared_initializer` parameter + `if shared_initializer is None:` guard in `build_many`.** The merged `build_many` signature is `build_many(models, input_shapes, key, d_neuron=2, d_layer=2, d_spatial=2, shared_initializer=None, dummy_input_dtype=jnp.float32)`.
- Test: `metanca/tests/metanca/nn/test__tasknet_tiny_lm.py` (new)

**Interfaces:**
- Produces: `TaskNet.build`/`build_many` working for `TinyCausalLM` with int32 inputs AND unchanged for MLP/Conv/ResNet; used by Tasks 8–10.

- [ ] **Step 1: Write the failing integration test**

```python
# metanca/tests/metanca/nn/test__tasknet_tiny_lm.py
import jax
import jax.numpy as jnp

import metanca
from metanca.nn import TinyCausalLM


def _lm(vocab=512, d=32, heads=2):
    return TinyCausalLM(vocab_size=vocab, d_model=d, num_heads=heads,
                        num_layers=3, mlp_dim=4 * d, max_seq_len=64)


def test_tasknet_builds_for_tiny_lm():
    tn = metanca.TaskNet.build(
        model=_lm(), input_shape=(64,), key=jax.random.key(0),
        n_spatial_dims=0, d_neuron=10, d_layer=10, d_spatial=10,
        dummy_input_dtype=jnp.int32,
    )
    names = list(tn.iter_param_names())
    assert any(n.endswith(".embedding") for n in names)
    assert any(n.endswith(".scale") for n in names)
    assert any("self_attn_qkv" in n for n in names)
    # every param has a hidden state of matching leading shape
    for n in names:
        p = tn.get(n, "param"); h = tn.get(n, "hidden_state")
        assert h.shape[:-1] == p.shape, n
    # reset must also work (uses the same dtype/multi-role paths)
    tn2 = tn.reset(jax.random.key(1))
    assert set(tn2.iter_param_names()) == set(names)


def test_build_many_shared_initializer_still_works_for_mlp():
    # regression: the public shared_initializer path must survive the merge
    from metanca.nn import MultiLayerPerceptron
    m = MultiLayerPerceptron(layer_specs=[16, 10])
    tns = metanca.TaskNet.build_many(models=[m], input_shapes=[(784,)],
                                     key=jax.random.key(0), d_neuron=10,
                                     d_layer=10, d_spatial=10)
    assert tns[0].hidden_dim == 30
```

- [ ] **Step 2: Run to confirm FAIL** (`dummy_input_dtype` unexpected kwarg).
- [ ] **Step 3: Apply the merge.** Work with both files open. The public file's `build_many` shared-initializer block and the monorepo's `dummy_input_dtype`/first-pass block must coexist: when `shared_initializer is None`, the first pass probes each model **with `dummy_input_dtype`**. If any listed delta is ambiguous against the actual code, STOP and report NEEDS_CONTEXT quoting the conflicting hunks — do not guess.
- [ ] **Step 4: Tests + full regression (both packages)**

```bash
cd metanca && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false ../.venv/bin/python -m pytest tests/metanca/nn/test__tasknet_tiny_lm.py -q && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false ../.venv/bin/python -m pytest -q
cd ../metanca_training && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false ../.venv/bin/python -m pytest tests/metanca_training/scaling/test__arch_grid.py tests/metanca_training/scaling/test__grid_initializer.py tests/metanca_training/scaling/test__build_many_shared_initializer.py -q
```
Expected: all PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
git add metanca/src metanca/tests
git commit -m "feat: TaskNet supports TinyCausalLM (multi-role params, int32 probes, constvar identity matching, bwd graphs)"
```

---

### Task 5: Shakespeare data + 4 BPE tokenizers

**Files:**
- Create: `metanca_training/data/shakespeare/input.txt` (copy from `/home/dan/Projects/monorepo/metanca_training/data/shakespeare/input.txt`)
- Create: `metanca_training/data/shakespeare/train_tokenizer.py` (copy from monorepo, then parameterize `vocab_size` via argparse if not already)
- Create (generated): `metanca_training/data/shakespeare/shakespeare_{512,1024,2048,4096}_bpe.{model,vocab}` + `metanca_training/data/shakespeare/chunking.json`
- Create: `metanca_training/data/shakespeare/compute_chunking.py` (new, below)

**Interfaces:**
- Produces: 4 tokenizer artifacts; `chunking.json` = `{"chunk_bytes": <int>, "context_length": 64, "vocabs": [512,1024,2048,4096], "p95_tokens": {"512": <int>, ...}}` where `chunk_bytes` is the largest byte size such that ≥95% of chunks tokenize to ≤63 tokens under vocab-512 (63 not 64: inputs/targets are ids[:-1]/ids[1:]).

- [ ] **Step 1: Copy input.txt + train_tokenizer.py; train the 4 tokenizers**

```bash
cd metanca_training/data/shakespeare
for v in 512 1024 2048 4096; do ../../../.venv/bin/python train_tokenizer.py --vocab-size $v --model-prefix shakespeare_${v}_bpe; done
```
(Adapt flags to the script's actual interface — keep `byte_fallback`, pad/eos ids as the monorepo's script sets them. If the script hardcodes 10000, add `--vocab-size`/`--model-prefix` argparse options.)

- [ ] **Step 2: Write `compute_chunking.py`**

```python
# metanca_training/data/shakespeare/compute_chunking.py
"""Pick chunk_bytes: largest byte span whose vocab-512 tokenization fits 63 tokens
for >=95% of chunks; record per-vocab P95 token counts."""
import json
import numpy as np
import sentencepiece as spm

TEXT = open("input.txt", "rb").read().decode("utf-8")
VOCABS = [512, 1024, 2048, 4096]
CONTEXT = 64

sps = {v: spm.SentencePieceProcessor(model_file=f"shakespeare_{v}_bpe.model") for v in VOCABS}

def p95_tokens(chunk_bytes, sp):
    chunks = [TEXT[i:i + chunk_bytes] for i in range(0, len(TEXT), chunk_bytes)]
    return float(np.percentile([len(sp.encode(c)) for c in chunks], 95))

chunk_bytes = next(b for b in range(200, 40, -8) if p95_tokens(b, sps[512]) <= CONTEXT - 1)
out = {"chunk_bytes": chunk_bytes, "context_length": CONTEXT, "vocabs": VOCABS,
       "p95_tokens": {str(v): p95_tokens(chunk_bytes, sps[v]) for v in VOCABS}}
json.dump(out, open("chunking.json", "w"), indent=2)
print(out)
```

- [ ] **Step 3: Run it**: `../../../.venv/bin/python compute_chunking.py` → `chunking.json` written, `chunk_bytes` between 48 and 200, all `p95_tokens` ≤ 63.
- [ ] **Step 4: Commit** (artifacts are small — commit them; they're study inputs)

```bash
git add metanca_training/data/shakespeare
git commit -m "feat: Shakespeare corpus + 4 BPE tokenizers (512..4096) + chunking calibration"
```

---

### Task 6: Multi-tokenizer collate data pipeline

**Files:**
- Create: `metanca_training/src/metanca_training/lm_data.py`
- Test: `metanca_training/tests/metanca_training/test__lm_data.py`

**Interfaces:**
- Produces:
  - `load_multi_vocab_shakespeare(data_dir: str, batch_size: int = 8, val_split: float = 0.1) -> MultiVocabLM` where `MultiVocabLM` is a NamedTuple: `train: dict[int, Batch]`, `val: dict[int, Batch]`, `factors: dict[int, dict[str, float]]` (`factors[v] = {"train": tokens/byte, "val": tokens/byte}`), `context_length: int`, `vocabs: tuple[int, ...]`. `Batch = (inputs, targets, mask)` int32/int32/bool arrays shaped `[n_batches, batch_size, context]` / same / `[..., context, 1]`; **`n_batches` identical across vocabs** (same chunk stream).
  - Pure helper `collate_chunks(chunks: list[str], sps: dict[int, Any], context: int, pad_ids: dict[int, int]) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]` — tokenize each chunk with every tokenizer; `inputs = ids[:-1]`, `targets = ids[1:]`, truncate to `context`, pad with `pad_id`, mask = positions with a real (non-pad) target.

- [ ] **Step 1: Write failing tests** (pure collate on a fake tokenizer, no sentencepiece needed)

```python
# metanca_training/tests/metanca_training/test__lm_data.py
import numpy as np
from metanca_training.lm_data import collate_chunks


class FakeSP:
    """chars -> byte values; deterministic, no external model."""
    def encode(self, s): return [min(b, 250) for b in s.encode("utf-8")]


def test_collate_shapes_and_alignment():
    chunks = ["hello world", "hi"]
    out = collate_chunks(chunks, {300: FakeSP()}, context=8, pad_ids={300: 251})
    X, Y, M = out[300]
    assert X.shape == (2, 8) and Y.shape == (2, 8) and M.shape == (2, 8, 1)
    ids = FakeSP().encode("hello world")
    assert list(X[0]) == ids[:-1][:8]           # inputs = ids[:-1], truncated
    assert list(Y[0]) == ids[1:][:9][:8]        # targets = ids[1:]
    n_valid = len(FakeSP().encode("hi")) - 1    # "hi" -> 2 ids -> 1 scored position
    assert M[1, :, 0].sum() == n_valid
    assert (X[1, n_valid:] == 251).all()        # padded with pad_id


def test_collate_same_chunks_all_vocabs():
    out = collate_chunks(["abc", "de"], {1: FakeSP(), 2: FakeSP()}, context=4,
                         pad_ids={1: 251, 2: 251})
    assert set(out) == {1, 2}
    assert out[1][0].shape == out[2][0].shape   # same n_chunks x context
```

- [ ] **Step 2: Run → FAIL** (module missing).
- [ ] **Step 3: Implement `lm_data.py`**

```python
# metanca_training/src/metanca_training/lm_data.py
"""Multi-tokenizer collate pipeline for the LLM scaling study.

The same text chunks are tokenized by every vocab's tokenizer (a collate over raw text),
so all architectures see identical underlying text per step. Shakespeare is small, so the
collate runs eagerly at load time into static-shape arrays. (A streaming corpus would use
grain with this collate as a per-batch transform.)"""

import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np


class MultiVocabLM(NamedTuple):
    train: dict[int, tuple]
    val: dict[int, tuple]
    factors: dict[int, dict[str, float]]
    context_length: int
    vocabs: tuple[int, ...]


def collate_chunks(chunks, sps: dict[int, Any], context: int, pad_ids: dict[int, int]):
    out = {}
    for v, sp in sps.items():
        pad = pad_ids[v]
        X = np.full((len(chunks), context), pad, dtype=np.int32)
        Y = np.full((len(chunks), context), pad, dtype=np.int32)
        M = np.zeros((len(chunks), context, 1), dtype=bool)
        for i, c in enumerate(chunks):
            ids = sp.encode(c)
            x, y = ids[:-1][:context], ids[1:][:context + 1][:context]
            n = min(len(x), len(y))
            X[i, :n] = x[:n]; Y[i, :n] = y[:n]; M[i, :n, 0] = True
        out[v] = (X, Y, M)
    return out


def _batchify(arr: np.ndarray, batch_size: int) -> np.ndarray:
    n = (len(arr) // batch_size) * batch_size          # drop ragged tail (shared across vocabs)
    return arr[:n].reshape(-1, batch_size, *arr.shape[1:])


def load_multi_vocab_shakespeare(data_dir: str, batch_size: int = 8, val_split: float = 0.1):
    import sentencepiece as spm
    d = Path(data_dir)
    cfg = json.loads((d / "chunking.json").read_text())
    context, vocabs = int(cfg["context_length"]), tuple(int(v) for v in cfg["vocabs"])
    text = (d / "input.txt").read_bytes().decode("utf-8")
    cb = int(cfg["chunk_bytes"])
    chunks = [text[i:i + cb] for i in range(0, len(text) - cb + 1, cb)]
    n_train = int(len(chunks) * (1.0 - val_split))
    sps = {v: spm.SentencePieceProcessor(model_file=str(d / f"shakespeare_{v}_bpe.model"))
           for v in vocabs}
    pad_ids = {v: sps[v].pad_id() for v in vocabs}

    train, val, factors = {}, {}, {}
    for split_name, split_chunks, store in (("train", chunks[:n_train], train),
                                            ("val", chunks[n_train:], val)):
        col = collate_chunks(split_chunks, sps, context, pad_ids)
        n_bytes = sum(len(c.encode("utf-8")) for c in split_chunks)
        for v, (X, Y, M) in col.items():
            store[v] = (_batchify(X, batch_size), _batchify(Y, batch_size),
                        _batchify(M, batch_size))
            factors.setdefault(v, {})[split_name] = float(M.sum()) / n_bytes
    return MultiVocabLM(train=train, val=val, factors=factors,
                        context_length=context, vocabs=vocabs)
```

- [ ] **Step 4: Tests pass**; also run a real-loader smoke: `cd metanca_training && JAX_PLATFORMS=cpu ../.venv/bin/python -c "from metanca_training.lm_data import load_multi_vocab_shakespeare as L; d=L('data/shakespeare'); print({v: d.train[v][0].shape for v in d.vocabs}, d.factors)"` — all vocabs share `n_batches`, factors ∈ (0.05, 2.0).
- [ ] **Step 5: Commit** — `git add metanca_training/src metanca_training/tests && git commit -m "feat: multi-tokenizer collate pipeline for Shakespeare (shared chunks, per-vocab batches)"`

---

### Task 7: Sparse-CE loss + LM metrics

**Files:**
- Modify: `metanca_training/src/metanca_training/_loss.py` — merge from `/home/dan/Projects/monorepo/metanca_training/src/metanca_training/_loss.py`: add `masked_sparse_softmax_cross_entropy` (via `optax.softmax_cross_entropy_with_integer_labels`) and the `_auto_loss_fn` dispatch (labels.ndim vs logits.ndim) as `compute_loss`'s default `loss_fn`.
- Create: `metanca_training/src/metanca_training/metrics/` — copy the monorepo package (`__init__.py`, `_bpb.py`, `_perplexity.py`, `_accuracy.py`, `_registry.py`, `_compute_metrics.py`) verbatim.
- Test: `metanca_training/tests/metanca_training/test__lm_loss_metrics.py`

**Interfaces:**
- Produces: `compute_loss(batch_x, batch_y_int32, mask, params, apply_fn)` works with integer targets + 3D logits; `metanca_training.metrics` exposes `bpb(loss)` (= loss/ln2, bits/TOKEN — Task 9 applies the tokens/bytes factor) and `perplexity(loss)`.

- [ ] **Step 1: Write failing tests**

```python
# metanca_training/tests/metanca_training/test__lm_loss_metrics.py
import numpy as np
import jax.numpy as jnp
from metanca_training._loss import compute_loss, masked_sparse_softmax_cross_entropy
from metanca_training.metrics import bpb, perplexity


def test_sparse_ce_matches_manual():
    logits = jnp.zeros((2, 3, 5))                      # uniform -> CE = ln(5)
    targets = jnp.array([[1, 2, 3], [0, 0, 0]], dtype=jnp.int32)
    mask = jnp.ones((2, 3, 1), dtype=bool)
    loss = masked_sparse_softmax_cross_entropy(logits, targets, mask)
    assert np.isclose(float(loss), np.log(5), atol=1e-5)


def test_compute_loss_dispatches_sparse():
    apply_fn = lambda p, x: jnp.zeros((x.shape[0], x.shape[1], 7))
    x = jnp.zeros((2, 4), dtype=jnp.int32)
    y = jnp.zeros((2, 4), dtype=jnp.int32)
    m = jnp.ones((2, 4, 1), dtype=bool)
    assert np.isclose(float(compute_loss(x, y, m, {}, apply_fn)), np.log(7), atol=1e-5)


def test_bpb_and_perplexity():
    assert np.isclose(float(bpb(jnp.log(2.0))), 1.0)
    assert np.isclose(float(perplexity(jnp.array(0.0))), 1.0)
```

- [ ] **Step 2: FAIL** (imports missing). **Step 3:** apply merge + copy (adjust `bpb`/`perplexity` import path to whatever the copied `metrics/__init__.py` exports). **Step 4:** tests pass + `metanca_training` scaling tests still green. **Step 5: Commit** `feat: sparse CE loss + LM metrics package (bpb, perplexity)`.

---

### Task 8: LLM grid module

**Files:**
- Create: `metanca_training/src/metanca_training/scaling/llm_grid.py`
- Test: `metanca_training/tests/metanca_training/scaling/test__llm_grid.py`

**Interfaces:**
- Produces:
  - `LLMArch = NamedTuple("LLMArch", [("d_model", int), ("num_heads", int), ("vocab", int)])`
  - `D_MODELS=(32,48,64,96,128,192,256)`, `HEADS=(2,4,8)`, `VOCABS=(512,1024,2048,4096)`, `HEAD_DIM_RANGE=(8,64)`, `NUM_LAYERS=3`, `N_VAL_LLM=14`
  - `build_llm_grid() -> list[LLMArch]` (68, deterministic order)
  - `llm_arch_id(a) -> str` = `f"llm_d{a.d_model}_h{a.num_heads}_v{a.vocab}"`
  - `build_tiny_lm(a, context_length) -> TinyCausalLM` (`num_layers=3, mlp_dim=4*d_model, max_seq_len=context_length`)
  - `split_llm_grid(seed) -> (pool, val)` and `sample_llm_subset(pool, t, seed)` — reuse `arch_grid._shuffled`.

- [ ] **Step 1: Failing tests**

```python
# metanca_training/tests/metanca_training/scaling/test__llm_grid.py
from metanca_training.scaling.llm_grid import (
    D_MODELS, HEADS, VOCABS, N_VAL_LLM, LLMArch,
    build_llm_grid, llm_arch_id, build_tiny_lm, split_llm_grid, sample_llm_subset,
)


def test_grid_size_and_validity():
    grid = build_llm_grid()
    assert len(grid) == 68 and len(set(grid)) == 68
    for a in grid:
        hd = a.d_model // a.num_heads
        assert a.d_model % a.num_heads == 0 and 8 <= hd <= 64
        assert a.d_model in D_MODELS and a.num_heads in HEADS and a.vocab in VOCABS


def test_combo_count_per_dmodel():
    combos = {(a.d_model, a.num_heads) for a in build_llm_grid()}
    per_d = {d: sum(1 for c in combos if c[0] == d) for d in D_MODELS}
    assert per_d == {32: 2, 48: 2, 64: 3, 96: 3, 128: 3, 192: 2, 256: 2}


def test_split_and_sample_deterministic():
    pool, val = split_llm_grid(seed=7)
    assert len(val) == N_VAL_LLM and len(pool) == 54
    assert set(pool).isdisjoint(val) and set(pool) | set(val) == set(build_llm_grid())
    assert split_llm_grid(seed=7) == (pool, val)
    sub = sample_llm_subset(pool, 8, seed=3)
    assert len(sub) == 8 and set(sub) <= set(pool)
    assert sample_llm_subset(pool, 8, seed=3) == sub


def test_arch_id_and_model():
    a = LLMArch(64, 4, 1024)
    assert llm_arch_id(a) == "llm_d64_h4_v1024"
    m = build_tiny_lm(a, context_length=64)
    assert (m.d_model, m.num_heads, m.num_layers, m.mlp_dim, m.vocab_size,
            m.max_seq_len) == (64, 4, 3, 256, 1024, 64)
```

- [ ] **Step 2: FAIL.** **Step 3: Implement**

```python
# metanca_training/src/metanca_training/scaling/llm_grid.py
"""LLM (TinyCausalLM) architecture grid for the Shakespeare scaling study."""
from typing import NamedTuple

from metanca.nn import TinyCausalLM

from .arch_grid import _shuffled

D_MODELS: tuple[int, ...] = (32, 48, 64, 96, 128, 192, 256)
HEADS: tuple[int, ...] = (2, 4, 8)
VOCABS: tuple[int, ...] = (512, 1024, 2048, 4096)
HEAD_DIM_RANGE: tuple[int, int] = (8, 64)
NUM_LAYERS: int = 3
N_VAL_LLM: int = 14


class LLMArch(NamedTuple):
    d_model: int
    num_heads: int
    vocab: int


def build_llm_grid() -> list[LLMArch]:
    lo, hi = HEAD_DIM_RANGE
    return [LLMArch(d, h, v) for d in D_MODELS for h in HEADS for v in VOCABS
            if d % h == 0 and lo <= d // h <= hi]


def llm_arch_id(a: LLMArch) -> str:
    return f"llm_d{a.d_model}_h{a.num_heads}_v{a.vocab}"


def build_tiny_lm(a: LLMArch, context_length: int) -> TinyCausalLM:
    return TinyCausalLM(vocab_size=a.vocab, d_model=a.d_model, num_heads=a.num_heads,
                        num_layers=NUM_LAYERS, mlp_dim=4 * a.d_model,
                        max_seq_len=context_length)


def split_llm_grid(seed: int) -> tuple[list[LLMArch], list[LLMArch]]:
    shuffled = [LLMArch(*t) for t in _shuffled(build_llm_grid(), seed)]
    return shuffled[N_VAL_LLM:], shuffled[:N_VAL_LLM]


def sample_llm_subset(pool, t: int, seed: int) -> list[LLMArch]:
    if t > len(pool):
        raise ValueError(f"cannot sample T={t} from pool of {len(pool)}")
    return [LLMArch(*x) for x in _shuffled(pool, seed)[:t]]
```

- [ ] **Step 4: PASS + scaling suite green. Step 5: Commit** `feat: LLM (d_model, heads, vocab) grid — 68 archs, V=14 split`.

---

### Task 9: Per-arch batch streams in the train loop + LLM evaluator

**Files:**
- Modify: `metanca_training/src/metanca_training/_calculate_metanca_gradients.py` — `calculate_metanca_gradients(batch=...)` → `batches: Sequence[tuple]` (one per tasknet); the per-arch loop uses `batches[i]`. Multi-GPU variant: same change (it already loops archs).
- Modify: `metanca_training/src/metanca_training/_metanca_train_step.py` — thread `batches` through both step functions (replaces `batch`).
- Modify: `metanca_training/src/metanca_training/_train_metanca.py` — accept `train_batches` as EITHER the classic `(X, Y, M)` tuple OR `dict[int, (X, Y, M)]` + new kwarg `arch_keys: list | None = None` (per-model key into the dict; last entry = test model's key). Build the per-arch tuple each step: classic → `batches = tuple([batch] * n_archs)`; dict → `batches = tuple(dict_batches[k][batch_idx-slice] for k in arch_keys)`. Also: `promote_image_batch` applied ONLY when dtype is uint8/float; int32 passes through. `n_train_batches` from any dict entry (all equal). In-loop validation uses the test model's key for `val_batches`. `dummy_input_dtype=jnp.int32` passed to `build_many` when batches are int (compute from the dict's array dtype).
- Test: `metanca_training/tests/metanca_training/test__per_arch_batches.py`

**Interfaces:**
- Consumes: Task 4's `build_many(..., dummy_input_dtype=...)`, Task 6's `MultiVocabLM`, Task 8's grid.
- Produces: `train_metanca(train_batches, val_batches, *, cfg, models, test_model, shared_initializer=None, arch_keys=None) -> (params, training_vars)` — dict-mode for LLMs, tuple-mode unchanged for classification (all existing callers unaffected).

- [ ] **Step 1: Failing test** — tuple-mode regression is covered by the existing iris smoke (`test__train_metanca_returns.py`); write the dict-mode smoke:

```python
# metanca_training/tests/metanca_training/test__per_arch_batches.py
import numpy as np
import jax, jax.numpy as jnp
import wandb
from pathlib import Path
from hydra import compose, initialize_config_dir

from metanca_training import train_metanca
from metanca_training._hydra_configs import register_configs
from metanca_training.scaling.llm_grid import LLMArch, build_tiny_lm

CONFIG_DIR = str((Path(__file__).parents[2] / "configs").resolve())


def _fake_lm_batches(vocab, n_batches=2, bs=2, ctx=16, seed=0):
    r = np.random.default_rng(seed)
    X = r.integers(0, vocab, (n_batches, bs, ctx)).astype(np.int32)
    Y = r.integers(0, vocab, (n_batches, bs, ctx)).astype(np.int32)
    M = np.ones((n_batches, bs, ctx, 1), dtype=bool)
    return X, Y, M


def test_dict_mode_two_vocabs_smoke():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=[
            "dataset=fashion_mnist", "wandb=disabled",       # dataset fields unused in dict-mode
            "training.num_metaepochs=1", "training.update_step_scheduler_max_steps=1",
            "training.early_stopping_enabled=false", "checkpoint.run_name=pytest_lm_smoke",
            "logging.log_every_n_metaepochs=1", "dataset.input_shape=[16]",
        ])
    wandb.init(mode="disabled")
    tb = {512: _fake_lm_batches(512), 1024: _fake_lm_batches(1024, seed=1)}
    vb = {512: _fake_lm_batches(512, seed=2), 1024: _fake_lm_batches(1024, seed=3)}
    models = [build_tiny_lm(LLMArch(32, 2, 512), 16), build_tiny_lm(LLMArch(32, 2, 1024), 16)]
    test_model = build_tiny_lm(LLMArch(32, 4, 1024), 16)
    params, tvars = train_metanca(
        tb, vb, cfg=cfg, models=models, test_model=test_model,
        arch_keys=[512, 1024, 1024],
    )
    assert params is not None and hasattr(tvars, "test_tasknet")
```

- [ ] **Step 2: FAIL** (`arch_keys` unexpected / dict unsupported). **Step 3: Implement** the three-file change; keep every existing call site compiling (tuple-mode default). **Step 4:** new test + iris smoke + scaling tests all pass (CPU; the smoke is tiny but slow — minutes are fine). **Step 5: Commit** `feat: per-arch batch streams (dict-mode) in train loop for mixed-vocab pools`.

---

### Task 10: LLM evaluator + runner

**Files:**
- Create: `metanca_training/src/metanca_training/scaling/evaluate_llm_pool.py` — mirror `evaluate_pool.py`: for each `(LLMArch, split_label)` build the tasknet (`build_tiny_lm`, `dummy_input_dtype=jnp.int32`, shared initializer from `training_vars.test_tasknet`), run `metanca_validation_step` on **that arch's vocab val batches** `n_init_samples` times, and emit rows `{arch_id, d_model, num_heads, vocab, split, val_loss_mean/std, val_ppl_mean/std, val_bpb_mean/std}` with `ppl = exp(loss)`, `bpb = loss * factors[vocab]["val"] / ln(2)`; keep `skip_ids`/`on_row`.
- Create: `metanca_training/scripts/scaling/run_llm_scaling.py` — clone of `run_scaling.py` with: grid calls → `llm_grid` (`SPLIT_SEED` same constant), data → `load_multi_vocab_shakespeare("metanca_training/data/shakespeare", batch_size=8)`, models → `build_tiny_lm(arch, ctx)`, `arch_keys=[a.vocab for a in train_archs] + [val_archs[0].vocab]`, `shared_initializer` from Task 10's provisioning (below), results under `results/scaling/llm/`, wandb run `scaling_llm_T{T}_rep{rep}` + `config.study="llm_shakespeare"`, `--metaepochs` default 1200 (probe-gated), eval via `evaluate_llm_pool`.
- Initializer provisioning (inside `run_llm_scaling.py`): build the shared initializer by probing the LARGEST arch — `metanca.TaskNet.build(model=build_tiny_lm(LLMArch(256, 4, 4096), ctx), ..., dummy_input_dtype=jnp.int32)` and reuse `(tn.hidden_state_initializer, tn.hidden_dim)` as `shared_initializer` for `train_metanca`. Add a coverage test asserting the grid's extreme archs build under it.
- Test: `metanca_training/tests/metanca_training/scaling/test__evaluate_llm_pool.py` (structure mirrors `test__evaluate_pool.py`: 1 train arch `LLMArch(32,2,512)` + 1 val arch `LLMArch(48,2,1024)`, fake batches from Task 9's helper via a tiny conftest util or duplicated helper, `n_update_steps=1`, `n_init_samples=1`; asserts row keys incl. `val_bpb_mean`, skip_ids/on_row behavior) and `test__llm_initializer_coverage.py` (largest-arch initializer builds `LLMArch(32,2,512)` and `LLMArch(256,8,4096)` tasknets without error).

**Interfaces:**
- Consumes: everything above.
- Produces: `evaluate_llm_pool(*, training_vars, local_rule_params, archs: list[tuple[LLMArch, str]], val_batches: dict[int, tuple], factors: dict, cfg, n_update_steps=10, n_init_samples=5, rand_key, skip_ids=None, on_row=None) -> list[dict]`; runnable `run_llm_scaling.py --T n --rep k [--smoke]` with the standard durable-run contract (`--ablation` is fixed to `llm` internally).

- [ ] **Step 1:** failing tests (as described; keep them CPU, tiny dims). **Step 2:** FAIL. **Step 3:** implement evaluator + runner (+ smoke mode: `--smoke` → metaepochs=2, 1 init, val batches truncated to 1). **Step 4:** tests pass; `--smoke` run completes end-to-end on CPU: `cd <worktree> && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python metanca_training/scripts/scaling/run_llm_scaling.py --T 2 --rep 0 --results-dir results/_llmsmoke --smoke` → `results/_llmsmoke/llm/T2_rep0.jsonl` has 2+14=16 rows + `.done`; re-run short-circuits. Clean up `results/_llmsmoke` + its checkpoint dir. **Step 5: Commit** `feat: LLM per-arch evaluator + durable scaling runner`.

---

### Task 11: Adam LLM baselines

**Files:**
- Create: `metanca_training/scripts/scaling/run_llm_adam_baselines.py` — analog of `run_adam_baselines.py`: for each of the 68 grid archs, plain-Adam train `build_tiny_lm(arch, ctx)` on its vocab's train batches to convergence (early stopping on val loss, patience 10, min_delta 1e-4, max_epochs 500, lr 1e-3), record `{val_loss, val_ppl, val_bpb, epochs, d_model, num_heads, vocab}` keyed by `llm_arch_id` → `--out results/adam_baselines_llm_shakespeare.json`; `--num-shards/--shard-index` for 2-GPU sharding; `--smoke` = first 2 archs, max_epochs 2. Reference for the Adam train-step shape: `/home/dan/Projects/monorepo/regular_network_baselines/scripts/train_tiny_lm_shakespeare.py` (adapt to `lm_data` loader + `masked_sparse_softmax_cross_entropy`).

**Interfaces:** consumes Tasks 6–8. Produces the baseline JSON table for plotting.

- [ ] **Step 1:** implement (script task — no unit test; verified by smoke). **Step 2:** CPU smoke: `--smoke --out results/_adam_llm_smoke.json` → 2 entries with the 7 keys; delete artifact. **Step 3: Commit** `feat: Adam-to-convergence LLM baselines over the 68-arch grid`.

---

### Task 12: Timing probe (controller-run on GPU)

**Files:**
- Create: `metanca_training/scripts/scaling/llm_probe.py` — measure, for archs `LLMArch(32,2,512)`, `LLMArch(128,4,2048)`, `LLMArch(256,4,4096)`: TaskNet build time, train-step compile time and steady s/batch at `n_update_steps ∈ {1, 10}` (constant schedule, T=1 each), plus one T=8 mixed-vocab pool steady rate; print projected per-run hours for the T grid at `--metaepochs 1200` given `n_batches` from the real loader; print peak device memory.

- [ ] **Step 1:** implement (structure follows `memory_probe.py`; uses dict-mode batches). Parse+import check only — the controller runs it on a free GPU and records numbers in `README_scaling.md`. **Step 2: Commit** `feat: LLM timing probe (grid corners, mixed-vocab pool)`.
- [ ] **Step 3 (controller):** run probe on a free 5090; **present projections to the user for budget sign-off. HARD GATE: no sweep launches until approved.**

---

### Task 13: Plotting for LLM metrics + final verification

**Files:**
- Modify: `metanca_training/src/metanca_training/scaling/plotting.py` — extend `_MEAN_KEY`/`_LABEL` with `{"bpb": "val_bpb_mean", "ppl": "val_ppl_mean"}` entries and labels ("validation bits/byte", "validation perplexity"); no other logic changes (T-axis/boxes/scatter are metric-agnostic).
- Modify: `metanca_training/scripts/scaling/plot_scaling.py` — add `--metrics` flag (default `loss,acc`; the LLM study passes `loss,bpb,ppl`).
- Test: extend `metanca_training/tests/metanca_training/scaling/test__plot_scaling.py` with a bpb fixture row asserting `aggregate(rows, "bpb")` reads `val_bpb_mean`.

- [ ] **Step 1:** failing test → implement → pass. **Step 2:** full CPU test pass of both packages (regression gate). **Step 3:** update `README_scaling.md` with the LLM study run order (tokenizers → probe → gate → `scheduler.py --ablation llm` equivalent commands). **Step 4: Commit** `feat: bpb/ppl plotting; LLM study docs`.

---

## Execution notes (controller)

- Tasks 2–4 are the risky merges: dispatch with sonnet, require the regression gate in every report; Task 4's implementer must quote conflicting hunks and stop rather than guess.
- Tasks 5, 11, 12 are script/artifact tasks verified by smoke; cheap models (haiku) suffice for 5.
- The scheduler (`scheduler.py`) is reused for the sweep by pointing its launch command at `run_llm_scaling.py` — a 10-line change deferred until after the probe gate (do not pre-build).
- RunPod: clone the `llm-scaling` branch to `/workspace/meta-nca-llm` with its own venv + tokenizer artifacts before the sweep; local GB10 runs small-T. All post-probe.
