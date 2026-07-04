# LLM Architecture-Scaling Study (Shakespeare) — Design

Status: approved pending user review (2026-07-04)
Branch: `llm-scaling` (worktree `.worktrees/llm-scaling`; the MLP study keeps the main checkout)

## Goal

Extend the architecture-scaling study from MLPs to language models: port the monorepo's
weight-level `TinyCausalLM` (GPT-style) MetaNCA support into the public repo, then run the
same T-scaling protocol — meta-train one local rule on T architectures, evaluate per-arch
performance on the T train archs and a fixed held-out val-arch set — on **Shakespeare**,
to test whether an architecture scaling law exists for language modeling.

Performance is measured on the data validation split as **val_loss** (mean token CE, nats),
**val_perplexity** (`exp(loss)`), and **val_bpb** (bits per byte) in place of the MLP study's
val accuracy/loss. **bpb is the only metric comparable across vocab sizes** and is the
headline cross-arch metric; loss/perplexity are reported per arch and compared only within a
vocab group.

## Architecture grid

`TinyCausalLM(vocab_size, d_model, num_heads, num_layers=3, mlp_dim=4*d_model,
max_seq_len=context_length)` — fused-QKV pre-norm attention, RMSNorm, GELU MLP, sinusoidal
(non-learned) positions, no weight tying, bias-free denses.

Three axes, fixed depth 3:

- `d_model ∈ {32, 48, 64, 96, 128, 192, 256}`
- `num_heads ∈ {2, 4, 8}`, valid iff `head_dim = d_model/num_heads` is an integer in `[8, 64]`
- `vocab ∈ {512, 1024, 2048, 4096}` (SentencePiece BPE retrained per size)

Valid `(d_model, heads)` combos: 32→{2,4}, 48→{2,4}, 64→{2,4,8}, 96→{2,4,8}, 128→{2,4,8},
192→{4,8}, 256→{4,8} = **17 combos × 4 vocabs = 68 architectures**.

Param range: ~70k (`d=32, V=512`) to ~4.5M (`d=256, V=4096`; ≈2.1M embedding+lm_head +
≈2.36M blocks). Arch id convention: `llm_d{d_model}_h{heads}_v{vocab}`.

Split and sweep (mirroring the MLP study):

- Hold out **V = 14** archs (~20%) with a fixed split seed; training pool = 54.
- **T ∈ {1, 2, 4, 8, 16, 32}**, **3 reps**, independent T-subsets per rep
  (`seed = 1000·T + rep` convention), rep-major scheduling with the existing
  scheduler/head-start/stall machinery. T values may be trimmed after the probe (below).

## Data pipeline (multi-tokenizer collate)

- Corpus: `data/shakespeare/input.txt` (copied from monorepo, ~1.1 MB).
- Four SentencePiece BPE tokenizers retrained at vocab 512/1024/2048/4096 using the
  monorepo's `train_tokenizer.py` (byte_fallback=true, pad/eos ids as there); artifacts
  committed under `metanca_training/data/shakespeare/`.
- **Shared-text chunking:** the text is split into fixed byte-span chunks sized so the
  finest tokenizer (512) fits `context_length = 64` tokens for ≥95% of chunks (chunk byte
  size measured empirically at tokenizer build time; rare overflows truncated, truncation
  rate logged). Chunks are the unit of batching: a batch of chunks is tokenized by **all
  four tokenizers** (the collate), producing `batches[vocab] = (inputs, targets, mask)`
  with identical `n_batches` and `batch_size` across vocabs — every arch sees the same
  underlying text every step, tokenized its own way.
- Eager materialization: Shakespeare is small and static, so the collate runs once at load
  time into static-shape arrays (jit-friendly; no per-step CPU tokenization). Grain is the
  designated path if a streaming corpus (e.g. FineWeb) is added later — out of scope now.
- Contiguous 90/10 train/val split over chunks with a fixed data key shared by every run
  and the baselines (one dataset, no per-run resplitting — chunk stream is deterministic).
- `batch_size = 8` chunks (monorepo convention; probe may revise), `context_length = 64`.
- Per split and vocab, record `N_scored_tokens` and `N_bytes` →
  `bpb = mean_token_CE_nats × (N_scored_tokens / N_bytes) / ln(2)`.

## Port from monorepo (`/home/dan/Projects/monorepo`)

**Copy (new files):**
- `metanca/src/metanca/nn/_tiny_causal_lm.py` (+ its test) and `TinyCausalLM` exports.
- `metanca/src/metanca/neighbors/_message_kernels/{_elementwise_nonbarrier_op,_add,_mul,_sub}.py`
  — includes the `shared_trailing_channel` elementwise case required by residual streams and
  the backward slice-strategy registrations.
- `metanca_training` metrics package (`metrics/`: bpb, perplexity, registry, compute), the
  sparse-CE loss additions (`masked_sparse_softmax_cross_entropy`, `_auto_loss_fn` dispatch
  in `compute_loss`), task resolution (`_tasks.py` concept), LM loaders
  (`sp_tokenize_text`, `prepare_causal_lm_batches`, `get_lm_text_datasets` — adapted to the
  multi-tokenizer collate), dataset/model configs, and the Adam tiny-LM baseline script.
- `sentencepiece` added to `metanca_training/requirements.in`.

**Adapt (careful merges — public repo has diverged):**
- `metanca/src/metanca/nn/_tasknet.py`: add `dummy_input_dtype` (int32 token inputs),
  multi-role param support (`kernel`/`embedding`/`scale`/`bias` all get hidden states + PEs;
  `primary_param_priority`), identity-based constvar↔param matching (the sinusoidal PE bakes
  non-param constants into the jaxpr), forward/backward `convert_parameter_graph` split, and
  the `_fix_conv_dense_boundary_pe` kernel-existence guard — **while preserving the public
  `shared_initializer` parameter on `build_many`** (the scaling machinery depends on it; the
  monorepo removed it and that removal must NOT be carried over).
- `metanca/src/metanca/neighbors/_convert_parameter_graph.py`: `slice_strategy` validation
  arg + `convert_parameter_graph_backward`.
- `metanca/src/metanca/hidden_state/_initializer_from_parameters.py`: layer-name regex and
  primary-shape selection extended to `embedding`/`scale`.
- `metanca/src/metanca/nn/_get_neighbors.py`: `embedding` (fwd/bwd) and `scale` cases.
- `metanca_training/_train_metanca.py`: only two LM-relevant edits — `dummy_input_dtype`
  computed from the dataset task and passed to `build_many`; `promote_image_batch` applied
  only to uint8 batches (int32 LM batches pass through).
- `_hydra_configs.py`: `TaskConfig` + LM dataset fields (`text_path`, `context_length`,
  `val_split`, …). Do not carry over the monorepo's unrelated changes (neuron-level config,
  early-stopping removal).

**Explicitly not ported:** neuron-level rule files (`_neuron_*`, `_weight_rule_net.py`,
`_update_tasknet_neuron.py`) — the monorepo's own audit marks them special-cased; FineWeb
streaming; the web demo.

**Regression gate for every core-library merge:** the public repo's existing suites
(`metanca` 76p/4s; `metanca_training` scaling tests) must stay green — the MLP study's
machinery keeps working on this branch.

## Scaling integration (new code)

- `scaling/llm_grid.py`: enumerate/split/sample the 68-arch grid
  (`build_llm_grid()`, `llm_arch_id()`, `build_tiny_lm(spec)`, reusing `split_grid`/
  `sample_subset` patterns; unit-tested counts: 17 combos, 68 archs, V=14/pool=54 split
  determinism).
- **Initializer provisioning:** one shared hidden-state initializer spanning the grid —
  neuron table covers `max(4096 (vocab), 4·256 (mlp))` = 4096 positions; layer table covers
  the traced layer count of a depth-3 `TinyCausalLM` (21 named layers: token_embed + 6/block
  × 3 + final_norm + lm_head), provisioned via the largest arch (`d=256, v=4096`) and
  verified by a coverage test over grid extremes (same pattern as
  `grid_hidden_state_initializer`; exact table sizes asserted from the trace, not hardcoded).
- **Train-loop change (the one structural edit):** `train_metanca` accepts per-arch batch
  streams — `batches: dict[vocab, (X, Y, M)]` plus `arch_vocab: list[int]`; the per-arch
  gradient loop feeds `batches[arch_vocab[i]][batch_idx]`. Shared `batch_idx` shuffling
  across vocabs (same chunks). Classification path (single shared batch) remains the
  default and unchanged.
- `scaling/evaluate_llm_pool.py`: mirrors `evaluate_arch_pool` — final params, 10 update
  steps, 5 random inits per arch, evaluated on the arch's own vocab val batches; rows carry
  `val_loss_mean/std`, `val_ppl_mean/std`, `val_bpb_mean/std` (+ arch fields: d_model,
  heads, vocab).
- `scripts/scaling/run_llm_scaling.py`: same durable per-run JSONL + `.done` + resume +
  wandb contract as `run_scaling.py` (results under `results/scaling/llm/`).
- Scheduler/orchestration: reuse `scheduler.py` (ablation name `llm`), same workers.
- Plotting: reuse `plotting.py` with metric keys extended (`bpb` primary two-panel boxplots
  + scatter; loss/ppl figures faceted or filtered by vocab group).

## Baselines

Adam-trained `TinyCausalLM` per grid arch (all 68), trained to convergence with early
stopping on val loss (patience 10, min-delta), recording `val_loss`, `val_ppl`, `val_bpb`,
epochs-to-best → `results/adam_baselines_llm_shakespeare.json`, sharded across the two 5090s
like the MLP baseline. Port of `regular_network_baselines/scripts/train_tiny_lm_shakespeare.py`
adapted to the shared data pipeline and grid.

## Feasibility gate (mandatory, before any full run)

Port + smoke first, then a timing probe before committing budget:
- Per-arch NCA step cost at grid corners (`d=32/v=512`, `d=256/v=4096`, and a middle arch),
  compile times, and steady s/metaepoch at T={1, 8} with the increment schedule.
- Deliverable: projected per-run and full-sweep wall-clock on the 3 available GPUs
  (GB10 + 2×5090), presented for approval with recommended (metaepochs, batch_size,
  T-ceiling, d_model-ceiling) adjustments. **No sweep launches before sign-off on the probe
  numbers.** Metaepoch budget target (pre-probe placeholder to validate): 1200 with
  increment rate 100, max 10 update steps — same shape as the MLP study.

## Execution

- Everything on `llm-scaling` (worktree). The MLP study finishes independently on
  `architecture-scaling-ablation`; GPUs are reused for LLM work as they free.
- wandb: same host project, runs named `scaling_llm_T{T}_rep{rep}` with
  `config.study = "llm_shakespeare"`.
- One-training-job-per-GPU discipline; persistent XLA compilation cache on both machines;
  RunPod worktree cloned at `/workspace/meta-nca-llm` (separate from the MLP checkout).
- Implementation: subagent-driven (fresh implementer per task + task review), TDD where
  logic is pure; core-library merge tasks carry the regression gate above.

## Out of scope

- Neuron-level local rule; FineWeb/streaming corpora (Grain adoption deferred with it);
  varying depth (`num_layers`) or context length as grid axes; weight tying; datasets other
  than Shakespeare; multi-GPU-per-run training (measured slower for small nets).
