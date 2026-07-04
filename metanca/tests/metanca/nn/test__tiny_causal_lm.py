import jax
import jax.numpy as jnp

from metanca.nn import TinyCausalLM


def test_tiny_causal_lm_returns_token_logits() -> None:
    model = TinyCausalLM(
        vocab_size=32,
        d_model=16,
        num_heads=4,
        num_layers=2,
        mlp_dim=32,
        max_seq_len=8,
    )
    token_ids = jnp.zeros((2, 8), dtype=jnp.int32)

    variables = model.init(jax.random.key(0), token_ids)
    logits = model.apply(variables, token_ids)

    assert logits.shape == (2, 8, 32)


def test_tiny_causal_lm_supports_bfloat16_parameters() -> None:
    model = TinyCausalLM(
        vocab_size=32,
        d_model=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        max_seq_len=8,
        param_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
    )
    token_ids = jnp.zeros((1, 8), dtype=jnp.int32)

    variables = model.init(jax.random.key(0), token_ids)
    embedding = variables["params"]["token_embed"]["embedding"]

    assert embedding.dtype == jnp.bfloat16


def test_tiny_causal_lm_uses_split_qkv_projections() -> None:
    model = TinyCausalLM(
        vocab_size=32,
        d_model=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        max_seq_len=8,
    )
    token_ids = jnp.zeros((1, 8), dtype=jnp.int32)

    variables = model.init(jax.random.key(0), token_ids)
    block_params = variables["params"]["blocks_0"]

    assert block_params["self_attn_qkv"]["kernel"].shape == (16, 48)
    assert "self_attn_query" not in block_params


def test_tiny_causal_lm_co4_like_configuration_has_about_8m_parameters() -> None:
    model = TinyCausalLM(
        vocab_size=258,
        d_model=400,
        num_heads=8,
        num_layers=4,
        mlp_dim=1600,
        max_seq_len=256,
        param_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
    )
    token_ids = jnp.zeros((1, 8), dtype=jnp.int32)

    variables = model.init(jax.random.key(0), token_ids)
    param_count = sum(leaf.size for leaf in jax.tree_util.tree_leaves(variables["params"]))

    assert param_count == 7_890_000
