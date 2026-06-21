from __future__ import annotations

from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp


def _default_kernel(x: jax.Array) -> jax.Array:
    """Positive feature map used by linear attention (Performer-style)."""
    return jax.nn.elu(x) + 1.0


def _normalize_key_mask(mask: jax.Array, seq_len: int, dtype: jnp.dtype) -> jax.Array:
    """Normalize a key mask to shape (batch, seq, 1, 1) for broadcasting."""
    if mask.ndim == 2:
        if mask.shape[1] != seq_len:
            raise ValueError(f"mask seq_len mismatch: got {mask.shape[1]}, expected {seq_len}")
        return mask[..., None, None].astype(dtype)
    if mask.ndim == 3:
        if mask.shape[1] != seq_len or mask.shape[2] != 1:
            raise ValueError(
                f"mask shape mismatch: got {mask.shape}, expected (batch, {seq_len}, 1)"
            )
        return mask[..., None].astype(dtype)
    raise ValueError(f"mask must have rank 2 or 3, got {mask.ndim}")


def _reshape_pos_enc(pos_enc: jax.Array, num_heads: int, head_dim: int) -> jax.Array:
    if pos_enc.ndim == 4:
        if pos_enc.shape[-2:] != (num_heads, head_dim):
            raise ValueError(
                f"pos_enc shape mismatch: expected (..., {num_heads}, "
                f"{head_dim}) got {pos_enc.shape}"
            )
        return pos_enc
    if pos_enc.ndim == 3 and pos_enc.shape[-1] == num_heads * head_dim:
        return pos_enc.reshape(pos_enc.shape[0], pos_enc.shape[1], num_heads, head_dim)
    raise ValueError(
        "pos_enc must have shape (batch, seq, heads*head_dim) or (batch, seq, heads, head_dim), "
        f"got {pos_enc.shape}"
    )


def _apply_rotary_pos_emb(x: jax.Array, pos_enc: jax.Array) -> jax.Array:
    sin = pos_enc[..., ::2]
    cos = pos_enc[..., 1::2]

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos

    return jnp.stack((rot_even, rot_odd), axis=-1).reshape(x.shape)


def _layer_norm(
    vectors: jax.Array, gamma: jax.Array, beta: jax.Array, epsilon: float = 1e-6
) -> jax.Array:
    mean = jnp.mean(vectors, axis=-1, keepdims=True)
    var = jnp.var(vectors, axis=-1, keepdims=True)
    normalized = (vectors - mean) / jnp.sqrt(var + epsilon)
    return gamma * normalized + beta


class LinearAttention(nn.Module):
    """
    Multi-head linear attention with learnable Q/K/V projections.

    This module implements the kernelized attention:

        Attn(Q, K, V) = (phi(Q) (phi(K)^T V)) / (phi(Q) (phi(K)^T 1) + eps)

    where phi is a positive feature map (default: ELU + 1). This avoids forming an
    explicit attention matrix and scales linearly with sequence length.

    Args:
        num_heads: Number of attention heads.
        head_dim: Per-head projection dimension. Defaults to input dim / num_heads.
        out_features: Output feature size. Defaults to query feature size.
        dropout_rate: Dropout rate applied to the attention output.
        use_bias: Whether Dense projections use bias.
        kernel_fn: Positive feature map applied to Q and K.
        epsilon: Numerical stability constant for the denominator.
        dtype: Computation dtype (defaults to inputs).
        param_dtype: Parameter dtype.
    """

    num_heads: int
    head_dim: int | None = None
    out_features: int | None = None
    dropout_rate: float = 0.0
    use_bias: bool = True
    kernel_fn: Callable[[jax.Array], jax.Array] = _default_kernel
    epsilon: float = 1e-6
    dtype: jnp.dtype | None = None
    param_dtype: jnp.dtype = jnp.float32
    use_post_ffn: bool = True
    ffn_hidden_dim: int = 100
    ffn_use_bias: bool = True
    norm_epsilon: float = 1e-6

    @nn.compact
    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        *,
        key_mask: jax.Array | None = None,
        query_mask: jax.Array | None = None,
        query_pos_enc: jax.Array | None = None,
        key_pos_enc: jax.Array | None = None,
        deterministic: bool | None = None,
    ) -> jax.Array:
        """
        Apply linear attention.

        Args:
            query: [batch, q_len, q_dim]
            key: [batch, kv_len, k_dim]
            value: [batch, kv_len, v_dim]
            key_mask: Optional boolean mask for keys/values, shape [batch, kv_len] or
                [batch, kv_len, 1]. False entries are ignored.
            query_mask: Optional boolean mask for query positions, shape [batch, q_len].
                Masked queries are zeroed in the output.
            deterministic: If True, disables dropout.
        """
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("query/key/value must have shape [batch, seq, features]")
        if key.shape[1] != value.shape[1]:
            raise ValueError("key and value must have the same sequence length")

        head_dim = self.head_dim
        if head_dim is None:
            if query.shape[-1] % self.num_heads != 0:
                raise ValueError(f"query feature dim {query.shape[-1]} not divisible by num_heads")
            head_dim = query.shape[-1] // self.num_heads
        out_features = self.out_features or query.shape[-1]

        dense_kwargs = {
            "use_bias": self.use_bias,
            "dtype": self.dtype,
            "param_dtype": self.param_dtype,
        }
        q_proj = nn.DenseGeneral(
            features=(self.num_heads, head_dim),
            name="query",
            **dense_kwargs,
        )
        k_proj = nn.DenseGeneral(
            features=(self.num_heads, head_dim),
            name="key",
            **dense_kwargs,
        )
        v_proj = nn.DenseGeneral(
            features=(self.num_heads, head_dim),
            name="value",
            **dense_kwargs,
        )

        if deterministic is None:
            deterministic = not self.is_mutable_collection("dropout")

        key_inputs = key
        value_inputs = value
        if (not deterministic) and self.dropout_rate > 0.0:
            keep_prob = 1.0 - self.dropout_rate
            dropout_mask = jax.random.bernoulli(
                self.make_rng("dropout"), p=keep_prob, shape=(key.shape[0], key.shape[1], 1)
            )
            key_inputs = (key_inputs * dropout_mask) / keep_prob
            value_inputs = (value_inputs * dropout_mask) / keep_prob

        q = q_proj(query)  # [batch, q_len, heads, head_dim]
        k = k_proj(key_inputs)  # [batch, kv_len, heads, head_dim]
        v = v_proj(value_inputs)  # [batch, kv_len, heads, head_dim]

        q_phi = self.kernel_fn(q)
        k_phi = self.kernel_fn(k)

        if query_pos_enc is not None:
            q_phi = _apply_rotary_pos_emb(
                q_phi, _reshape_pos_enc(query_pos_enc, self.num_heads, head_dim)
            )
        if key_pos_enc is not None:
            k_phi = _apply_rotary_pos_emb(
                k_phi, _reshape_pos_enc(key_pos_enc, self.num_heads, head_dim)
            )

        if key_mask is not None:
            key_mask = _normalize_key_mask(key_mask, key.shape[1], k_phi.dtype)
            k_phi = k_phi * key_mask
            v = v * key_mask

        # Precompute global key/value summaries.
        k_phi_t_v = jnp.einsum("bshd,bshm->bhdm", k_phi, v)
        k_phi_sum = jnp.sum(k_phi, axis=1)  # [batch, heads, head_dim]

        numerator = jnp.einsum("bthd,bhdm->bthm", q_phi, k_phi_t_v)
        denominator = jnp.einsum("bthd,bhd->bth", q_phi, k_phi_sum) + self.epsilon
        attn_output = numerator / denominator[..., None]

        attn_output = attn_output.reshape(query.shape[0], query.shape[1], -1)
        output = nn.Dense(out_features, name="out", **dense_kwargs)(attn_output)

        if self.use_post_ffn:
            gamma_1 = self.param("gamma_1", nn.initializers.ones, (out_features,))
            beta_1 = self.param("beta_1", nn.initializers.zeros, (out_features,))
            gamma_2 = self.param("gamma_2", nn.initializers.ones, (out_features,))
            beta_2 = self.param("beta_2", nn.initializers.zeros, (out_features,))

            ffn_dense_kwargs = {**dense_kwargs, "use_bias": self.ffn_use_bias}
            norm_resid = _layer_norm(output + query, gamma_1, beta_1, epsilon=self.norm_epsilon)
            ff_hidden = nn.Dense(self.ffn_hidden_dim, name="ff_1", **ffn_dense_kwargs)(norm_resid)
            ff_hidden = jax.nn.relu(ff_hidden)
            ff_out = nn.Dense(out_features, name="ff_2", **ffn_dense_kwargs)(ff_hidden)
            output = _layer_norm(ff_out + norm_resid, gamma_2, beta_2, epsilon=self.norm_epsilon)

        if query_mask is not None:
            if query_mask.ndim != 2 or query_mask.shape[1] != query.shape[1]:
                raise ValueError(f"query_mask must have shape (batch, {query.shape[1]})")
            output = output * query_mask[..., None].astype(output.dtype)

        return output
