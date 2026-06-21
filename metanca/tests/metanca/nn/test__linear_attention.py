import os

import jax
import jax.numpy as jnp
import pytest

from metanca.nn import LinearAttention


def _dense_general(inputs: jax.Array, kernel: jax.Array, bias: jax.Array | None) -> jax.Array:
    output = jnp.einsum("btd,dhm->bthm", inputs, kernel)
    if bias is None:
        return output
    return output + bias


def _reference_linear_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    key_mask: jax.Array | None,
    kernel_fn,
    epsilon: float,
) -> jax.Array:
    q_phi = kernel_fn(q)
    k_phi = kernel_fn(k)
    if key_mask is not None:
        key_mask = key_mask[..., None, None].astype(k_phi.dtype)
        k_phi = k_phi * key_mask
        v = v * key_mask

    numerator = jnp.einsum("bthd,bshd,bshm->bthm", q_phi, k_phi, v)
    denominator = jnp.einsum("bthd,bshd->bth", q_phi, k_phi) + epsilon
    return numerator / denominator[..., None]


@pytest.mark.skipif(os.environ.get("RUN_LONG_TESTS", "no") == "no", reason="Skipping long tests")
def test_linear_attention_matches_reference() -> None:
    key = jax.random.key(0)
    key, q_key, k_key, v_key, init_key = jax.random.split(key, 5)

    batch, q_len, kv_len, features = 2, 3, 4, 8
    num_heads, head_dim = 2, 4

    query = jax.random.normal(q_key, (batch, q_len, features))
    key_values = jax.random.normal(k_key, (batch, kv_len, features))
    value = jax.random.normal(v_key, (batch, kv_len, features))
    key_mask = jnp.array([[1, 1, 0, 1], [1, 0, 0, 1]], dtype=bool)

    module = LinearAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        out_features=features,
        dropout_rate=0.0,
        use_post_ffn=False,
    )
    params = module.init(
        init_key,
        query,
        key_values,
        value,
        key_mask=key_mask,
        deterministic=True,
    )
    output = module.apply(
        params,
        query,
        key_values,
        value,
        key_mask=key_mask,
        deterministic=True,
    )

    weights = params["params"]
    q = _dense_general(query, weights["query"]["kernel"], weights["query"].get("bias"))
    k = _dense_general(key_values, weights["key"]["kernel"], weights["key"].get("bias"))
    v = _dense_general(value, weights["value"]["kernel"], weights["value"].get("bias"))
    ref_attn = _reference_linear_attention(
        q,
        k,
        v,
        key_mask,
        module.kernel_fn,
        module.epsilon,
    )
    ref_attn = ref_attn.reshape(batch, q_len, -1)
    ref_output = jnp.einsum("btd,df->btf", ref_attn, weights["out"]["kernel"])
    if "bias" in weights["out"]:
        ref_output = ref_output + weights["out"]["bias"]

    assert jnp.allclose(output, ref_output, atol=1e-5)


@pytest.mark.skipif(os.environ.get("RUN_LONG_TESTS", "no") == "no", reason="Skipping long tests")
def test_linear_attention_jittable_with_query_mask() -> None:
    key = jax.random.key(1)
    batch, seq_len, features = 1, 5, 6
    num_heads, head_dim = 3, 2

    inputs = jax.random.normal(key, (batch, seq_len, features))
    key_mask = jnp.array([[1, 1, 1, 0, 0]], dtype=bool)
    query_mask = jnp.array([[1, 1, 0, 0, 0]], dtype=bool)

    module = LinearAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        out_features=features,
        dropout_rate=0.0,
        use_post_ffn=False,
    )
    params = module.init(
        key,
        inputs,
        inputs,
        inputs,
        key_mask=key_mask,
        query_mask=query_mask,
        deterministic=True,
    )

    apply_fn = jax.jit(
        lambda p, x, km, qm: module.apply(
            p,
            x,
            x,
            x,
            key_mask=km,
            query_mask=qm,
            deterministic=True,
        )
    )
    output = apply_fn(params, inputs, key_mask, query_mask)

    assert jnp.allclose(output[:, 2:, :], 0.0)


@pytest.mark.skipif(os.environ.get("RUN_LONG_TESTS", "no") == "no", reason="Skipping long tests")
def test_linear_attention_dropout_changes_output() -> None:
    key = jax.random.key(2)
    batch, seq_len, features = 2, 4, 8
    num_heads, head_dim = 2, 4

    inputs = jax.random.normal(key, (batch, seq_len, features))
    module = LinearAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        out_features=features,
        dropout_rate=0.5,
        use_post_ffn=False,
    )
    params = module.init(key, inputs, inputs, inputs, deterministic=True)

    out_1 = module.apply(
        params,
        inputs,
        inputs,
        inputs,
        rngs={"dropout": jax.random.key(3)},
        deterministic=False,
    )
    out_2 = module.apply(
        params,
        inputs,
        inputs,
        inputs,
        rngs={"dropout": jax.random.key(4)},
        deterministic=False,
    )

    assert not jnp.allclose(out_1, out_2)
