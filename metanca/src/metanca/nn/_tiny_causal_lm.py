import flax.linen as nn
import jax
import jax.numpy as jnp


def _sinusoidal_position_encoding(seq_len: int, d_model: int, dtype: jnp.dtype) -> jax.Array:
    position = jnp.arange(seq_len, dtype=jnp.float32)[:, None]
    div_term = jnp.exp(jnp.arange(0, d_model, 2, dtype=jnp.float32) * (-jnp.log(10000.0) / d_model))
    pos_enc = jnp.zeros((seq_len, d_model), dtype=jnp.float32)
    pos_enc = pos_enc.at[:, 0::2].set(jnp.sin(position * div_term))
    pos_enc = pos_enc.at[:, 1::2].set(jnp.cos(position * div_term))
    return pos_enc.astype(dtype)


class TinyCausalLMBlock(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int
    compute_dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def _split_heads(self, x: jax.Array) -> jax.Array:
        batch_size, seq_len, _ = x.shape
        head_dim = self.d_model // self.num_heads
        return x.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: jax.Array) -> jax.Array:
        batch_size, _, seq_len, head_dim = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

    @nn.compact
    def __call__(self, x: jax.Array, causal_mask: jax.Array) -> jax.Array:
        attn_inputs = nn.RMSNorm(
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="attn_norm",
        )(x)
        qkv = nn.Dense(
            3 * self.d_model,
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="self_attn_qkv",
        )(attn_inputs)
        query, key, value = jnp.split(qkv, 3, axis=-1)

        query_heads = self._split_heads(query)
        key_heads = self._split_heads(key)
        value_heads = self._split_heads(value)
        head_dim = self.d_model // self.num_heads
        attn_scores = jnp.einsum("bhqd,bhkd->bhqk", query_heads, key_heads) / jnp.sqrt(head_dim)
        attn_scores = jnp.where(causal_mask, attn_scores, jnp.finfo(attn_scores.dtype).min)
        attn_weights = nn.softmax(attn_scores, axis=-1)
        attn_context = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, value_heads)
        attn_outputs = nn.Dense(
            self.d_model,
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="self_attn_out",
        )(self._merge_heads(attn_context))
        x = x + attn_outputs

        mlp_inputs = nn.RMSNorm(
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="mlp_norm",
        )(x)
        mlp_hidden = nn.Dense(
            self.mlp_dim,
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="mlp_in",
        )(mlp_inputs)
        mlp_hidden = nn.gelu(mlp_hidden)
        mlp_outputs = nn.Dense(
            self.d_model,
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="mlp_out",
        )(mlp_hidden)
        return x + mlp_outputs


class TinyCausalLM(nn.Module):
    vocab_size: int
    d_model: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    max_seq_len: int
    compute_dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, token_ids: jax.Array) -> jax.Array:
        if token_ids.ndim != 2:
            raise ValueError(f"Expected token_ids shape [batch, seq], got {token_ids.shape}")
        if token_ids.shape[1] > self.max_seq_len:
            raise ValueError(
                f"Sequence length {token_ids.shape[1]} exceeds max_seq_len {self.max_seq_len}"
            )

        token_embeddings = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="token_embed",
        )(token_ids)
        position_embeddings = _sinusoidal_position_encoding(
            token_ids.shape[1],
            self.d_model,
            self.compute_dtype,
        )[None, :, :]

        x = token_embeddings + position_embeddings
        causal_mask = nn.make_causal_mask(token_ids)

        for layer_idx in range(self.num_layers):
            x = TinyCausalLMBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                compute_dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                name=f"blocks_{layer_idx}",
            )(x, causal_mask)

        x = nn.RMSNorm(
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="final_norm",
        )(x)
        return nn.Dense(
            self.vocab_size,
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="lm_head",
        )(x)
