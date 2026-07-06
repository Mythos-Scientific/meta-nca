"""LLM (TinyCausalLM) architecture grid for the Shakespeare scaling study."""
from typing import NamedTuple

from metanca.nn import TinyCausalLM

from .arch_grid import _shuffled

D_MODELS: tuple[int, ...] = (32, 64, 96, 128)
HEADS: tuple[int, ...] = (4, 8, 16)
MLP_RATIOS: tuple[int, ...] = (1, 2, 4)
VOCABS: tuple[int, ...] = (10000,)  # fixed vocab: 10k-BPE Shakespeare tokenizer
HEAD_DIM_RANGE: tuple[int, int] = (8, 64)
NUM_LAYERS: int = 3
N_VAL_LLM: int = 5  # ~20% of the 24-arch grid


class LLMArch(NamedTuple):
    d_model: int
    num_heads: int
    vocab: int
    mlp_ratio: int


def build_llm_grid() -> list[LLMArch]:
    lo, hi = HEAD_DIM_RANGE
    return [LLMArch(d, h, v, r) for d in D_MODELS for h in HEADS for r in MLP_RATIOS
            for v in VOCABS if d % h == 0 and lo <= d // h <= hi]


def llm_arch_id(a: LLMArch) -> str:
    return f"llm_d{a.d_model}_h{a.num_heads}_r{a.mlp_ratio}_v{a.vocab}"


def build_tiny_lm(a: LLMArch, context_length: int) -> TinyCausalLM:
    return TinyCausalLM(vocab_size=a.vocab, d_model=a.d_model, num_heads=a.num_heads,
                        num_layers=NUM_LAYERS, mlp_dim=a.mlp_ratio * a.d_model,
                        max_seq_len=context_length)


def split_llm_grid(seed: int) -> tuple[list[LLMArch], list[LLMArch]]:
    shuffled = [LLMArch(*t) for t in _shuffled(build_llm_grid(), seed)]
    return shuffled[N_VAL_LLM:], shuffled[:N_VAL_LLM]


def sample_llm_subset(pool, t: int, seed: int) -> list[LLMArch]:
    if t > len(pool):
        raise ValueError(f"cannot sample T={t} from pool of {len(pool)}")
    return [LLMArch(*x) for x in _shuffled(pool, seed)[:t]]


# --- width-only study: heads and mlp_ratio fixed, d_model is the single axis ---
# Width is the one architecture axis the rule's neuron PEs actually encode (head count
# never changes parameter shapes), so this grid varies only d_model.
WIDTH_HEADS: int = 4          # head_dim = d/4 spans 8..64, satisfying HEAD_DIM_RANGE
WIDTH_MLP_RATIO: int = 4
WIDTH_RANGE: tuple[int, int] = (32, 256)
N_WIDTHS: int = 24
WIDTH_VAL_STRIDE: int = 3     # every 3rd width held out -> 8 val, 16 train


def build_width_grid() -> list[LLMArch]:
    """24 widths: linspace over WIDTH_RANGE rounded to multiples of 8 (keeps d % heads == 0
    and even head_dims); rounding preserves distinctness for this range/count."""
    lo, hi = WIDTH_RANGE
    ds = [int(round((lo + i * (hi - lo) / (N_WIDTHS - 1)) / 8)) * 8 for i in range(N_WIDTHS)]
    (v,) = VOCABS
    return [LLMArch(d, WIDTH_HEADS, v, WIDTH_MLP_RATIO) for d in ds]


def split_width_grid() -> tuple[list[LLMArch], list[LLMArch]]:
    """Deterministic interleaved split: indices 1, 4, ..., 22 are val, so held-out widths
    evenly span the range and val performance measures interpolation in width."""
    grid = build_width_grid()
    val = [a for i, a in enumerate(grid) if i % WIDTH_VAL_STRIDE == 1]
    train = [a for i, a in enumerate(grid) if i % WIDTH_VAL_STRIDE != 1]
    return train, val


def nested_width_subset(pool, t: int, rep_seed: int) -> list[LLMArch]:
    """Prefix of a per-rep shuffle: seed with the rep only (never T) so that within a rep
    the T-subsets are nested (T_i is a prefix of T_j for i < j)."""
    if t > len(pool):
        raise ValueError(f"cannot sample T={t} from pool of {len(pool)}")
    return [LLMArch(*x) for x in _shuffled(pool, rep_seed)[:t]]
