from metanca_training.scaling.arch_grid import (
    WIDTHS, N_VAL, MAX_HIDDEN_WIDTH, MAX_HIDDEN_LAYERS,
    enumerate_arch_widths, arch_id, arch_layer_specs,
    build_mlp, build_grid, split_grid, sample_subset,
)


def test_grid_maxima_bound_both_grids():
    # provisioning constants must upper-bound every arch in both grids
    for kind in ("fixed3", "varying"):
        for w in build_grid(kind):
            assert len(w) <= MAX_HIDDEN_LAYERS          # depth covered
            assert max(w) <= MAX_HIDDEN_WIDTH           # width covered
    assert MAX_HIDDEN_WIDTH == max(WIDTHS) and MAX_HIDDEN_LAYERS == 5


def test_enumerate_counts_per_depth():
    # C(len(WIDTHS)+d-1, d) with 4 widths
    assert len(enumerate_arch_widths([2])) == 10
    assert len(enumerate_arch_widths([3])) == 20
    assert len(enumerate_arch_widths([4])) == 35
    assert len(enumerate_arch_widths([5])) == 56


def test_enumerate_non_increasing_and_valid_widths():
    for w in enumerate_arch_widths([2, 3, 4, 5]):
        assert all(a >= b for a, b in zip(w, w[1:])), w
        assert all(x in WIDTHS for x in w), w


def test_enumerate_no_duplicates():
    archs = enumerate_arch_widths([2, 3, 4, 5])
    assert len(archs) == len(set(archs))


def test_build_grid_sizes():
    assert len(build_grid("fixed3")) == 20
    assert len(build_grid("varying")) == 121
    assert all(len(w) == 3 for w in build_grid("fixed3"))
    assert {len(w) for w in build_grid("varying")} == {2, 3, 4, 5}


def test_arch_id_and_layer_specs():
    assert arch_id((128, 64, 16)) == "d3_128-64-16"
    assert arch_layer_specs((128, 64, 16), 10) == [128, 64, 16, 10]


def test_split_grid_deterministic_disjoint():
    grid = build_grid("fixed3")
    pool_a, val_a = split_grid(grid, N_VAL["fixed3"], seed=0)
    pool_b, val_b = split_grid(grid, N_VAL["fixed3"], seed=0)
    assert val_a == val_b and pool_a == pool_b            # deterministic
    assert len(val_a) == 8 and len(pool_a) == 12
    assert set(pool_a).isdisjoint(set(val_a))
    assert set(pool_a) | set(val_a) == set(grid)
    _, val_c = split_grid(grid, N_VAL["fixed3"], seed=1)
    assert val_c != val_a                                 # seed changes split


def test_sample_subset_size_and_membership():
    pool, _ = split_grid(build_grid("fixed3"), N_VAL["fixed3"], seed=0)
    sub = sample_subset(pool, 5, seed=3)
    assert len(sub) == 5 and set(sub).issubset(set(pool))
    assert sample_subset(pool, 5, seed=3) == sub          # deterministic


def test_build_mlp_forward_shape():
    import jax, jax.numpy as jnp
    model = build_mlp((32, 32), n_classes=10)
    params = model.init(jax.random.key(0), jnp.zeros((4, 784)))
    out = model.apply(params, jnp.zeros((4, 784)))
    assert out.shape == (4, 10)
