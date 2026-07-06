from metanca_training.scaling.llm_grid import (
    D_MODELS, HEADS, MLP_RATIOS, VOCABS, N_VAL_LLM, WIDTH_HEADS, WIDTH_MLP_RATIO, LLMArch,
    build_llm_grid, llm_arch_id, build_tiny_lm, split_llm_grid, sample_llm_subset,
    build_width_grid, split_width_grid, nested_width_subset,
)


def test_grid_size_and_validity():
    grid = build_llm_grid()
    assert len(grid) == 24 and len(set(grid)) == 24
    for a in grid:
        hd = a.d_model // a.num_heads
        assert a.d_model % a.num_heads == 0 and 8 <= hd <= 64
        assert a.d_model in D_MODELS and a.num_heads in HEADS and a.vocab in VOCABS
        assert a.mlp_ratio in MLP_RATIOS


def test_combo_count_per_dmodel():
    combos = {(a.d_model, a.num_heads) for a in build_llm_grid()}
    per_d = {d: sum(1 for c in combos if c[0] == d) for d in D_MODELS}
    assert per_d == {32: 1, 64: 2, 96: 2, 128: 3}


def test_split_and_sample_deterministic():
    pool, val = split_llm_grid(seed=7)
    assert len(val) == N_VAL_LLM and len(pool) == 19
    assert set(pool).isdisjoint(val) and set(pool) | set(val) == set(build_llm_grid())
    assert split_llm_grid(seed=7) == (pool, val)
    sub = sample_llm_subset(pool, 8, seed=3)
    assert len(sub) == 8 and set(sub) <= set(pool)
    assert sample_llm_subset(pool, 8, seed=3) == sub


def test_width_grid():
    grid = build_width_grid()
    ds = [a.d_model for a in grid]
    assert len(ds) == 24 and len(set(ds)) == 24 and ds == sorted(ds)
    assert ds[0] == 32 and ds[-1] == 224
    for a in grid:
        assert a.num_heads == WIDTH_HEADS and a.mlp_ratio == WIDTH_MLP_RATIO
        assert a.d_model % 8 == 0 and 8 <= a.d_model // a.num_heads <= 64
        assert a.vocab in VOCABS


def test_width_split_interleaved():
    train, val = split_width_grid()
    grid = build_width_grid()
    assert len(train) == 16 and len(val) == 8
    assert [grid.index(a) for a in val] == [1, 4, 7, 10, 13, 16, 19, 22]
    assert set(train).isdisjoint(val) and set(train) | set(val) == set(grid)
    assert split_width_grid() == (train, val)


def test_width_subsets_nested_within_rep():
    train, _ = split_width_grid()
    for rep_seed in (0, 1, 2):
        prev = None
        for t in (1, 2, 4, 8, 16):
            sub = nested_width_subset(train, t, rep_seed=rep_seed)
            assert len(sub) == t and len(set(sub)) == t and set(sub) <= set(train)
            if prev is not None:
                assert sub[: len(prev)] == prev  # T_i is a prefix of T_j
            prev = sub
    # different reps get different orderings
    assert (nested_width_subset(train, 16, rep_seed=0)
            != nested_width_subset(train, 16, rep_seed=1))


def test_arch_id_and_model():
    a = LLMArch(64, 4, 10000, 2)
    assert llm_arch_id(a) == "llm_d64_h4_r2_v10000"
    m = build_tiny_lm(a, context_length=64)
    assert (m.d_model, m.num_heads, m.num_layers, m.mlp_dim, m.vocab_size,
            m.max_seq_len) == (64, 4, 3, 128, 10000, 64)
