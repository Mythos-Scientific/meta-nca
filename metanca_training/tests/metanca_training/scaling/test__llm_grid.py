from metanca_training.scaling.llm_grid import (
    D_MODELS, HEADS, VOCABS, N_VAL_LLM, LLMArch,
    build_llm_grid, llm_arch_id, build_tiny_lm, split_llm_grid, sample_llm_subset,
)


def test_grid_size_and_validity():
    grid = build_llm_grid()
    assert len(grid) == 51 and len(set(grid)) == 51
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
    assert len(val) == N_VAL_LLM and len(pool) == 41
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
