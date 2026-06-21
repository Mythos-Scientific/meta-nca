import flax.linen as nn
import jax
import jax.numpy as jnp

import metanca.functional as mfx

from ._linear_attention import LinearAttention
from ._multi_layer_perceptron import MultiLayerPerceptron


class LocalRuleNet(nn.Module):

    hidden_dim: int
    local_rule_layer_widths: tuple[int, ...]
    weight_transformer_dropout: float = 0.0
    n_spatial_dims: int = 2
    num_heads: int | None = None
    bias_linear_attn: bool = False
    bias_local_rule: bool = True

    def setup(self):
        state_dim = self.hidden_dim + 1
        num_heads = self.num_heads if self.num_heads is not None else (3 + self.n_spatial_dims)
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if self.hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by num_heads={num_heads}. "
                "Choose a compatible attention_num_heads or hidden state dimensions."
            )
        head_dim = self.hidden_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim={head_dim} must be even for rotary positional embeddings. "
                f"Need hidden_dim % (2 * num_heads) == 0, got hidden_dim={self.hidden_dim}, "
                f"num_heads={num_heads}."
            )

        self.attn_forward = LinearAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            out_features=state_dim,
            dropout_rate=self.weight_transformer_dropout,
            use_bias=self.bias_linear_attn,
            use_post_ffn=True,
            ffn_hidden_dim=100,
        )
        self.attn_backward = LinearAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            out_features=state_dim,
            dropout_rate=self.weight_transformer_dropout,
            use_bias=self.bias_linear_attn,
            use_post_ffn=True,
            ffn_hidden_dim=100,
        )

        self.local_rule_head = MultiLayerPerceptron(
            layer_specs=self.local_rule_layer_widths,
            activation="elu",
            final_layer_activation="tanh",
            final_layer_kernel_init=nn.initializers.zeros,
            final_layer_bias_init=nn.initializers.zeros,
            use_bias=self.bias_local_rule,
        )

    def __call__(
        self,
        focus_vectors_fwd: jax.Array,
        focus_vectors_bwd: jax.Array,
        neighbor_vectors_fwd: jax.Array,
        neighbor_vectors_bwd: jax.Array,
        focus_pos_enc_fwd: jax.Array | None,
        focus_pos_enc_bwd: jax.Array | None,
        neighbor_pos_enc_fwd: jax.Array | None,
        neighbor_pos_enc_bwd: jax.Array | None,
        param_mask_fwd_nv: jax.Array,
        param_mask_bwd_nv: jax.Array,
        num_cells_updated: int,
        param_shape: tuple[int, ...] | None = None,
        fwd_index_dim: int | None = None,
        fwd_stride: int | None = None,
        bwd_index_dim: int | None = None,
        bwd_stride: int | None = None,
        deterministic: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:

        if neighbor_vectors_fwd.size == 0:
            agg_fwd = jnp.zeros_like(focus_vectors_fwd)
        else:
            agg_fwd = self.attn_forward(
                focus_vectors_fwd,
                neighbor_vectors_fwd,
                neighbor_vectors_fwd,
                query_mask=param_mask_fwd_nv,
                query_pos_enc=focus_pos_enc_fwd,
                key_pos_enc=neighbor_pos_enc_fwd,
                deterministic=deterministic,
            )

        if neighbor_vectors_bwd.size == 0:
            agg_bwd = jnp.zeros_like(focus_vectors_bwd)
        else:
            agg_bwd = self.attn_backward(
                focus_vectors_bwd,
                neighbor_vectors_bwd,
                neighbor_vectors_bwd,
                query_mask=param_mask_bwd_nv,
                query_pos_enc=focus_pos_enc_bwd,
                key_pos_enc=neighbor_pos_enc_bwd,
                deterministic=deterministic,
            )

        # NOTE: No count division here. Linear attention already normalizes
        # via its denominator. Dividing again by neighbor count would
        # double-normalize and introduce architecture-dependent scaling,
        # preventing generalization to unseen architectures.

        # Transform agg_bwd from backward neuron view coordinates to forward
        # neuron view coordinates so it can be indexed with forward indices.
        # Without this, backward attention outputs are indexed with the wrong
        # coordinate system, especially at conv-dense boundaries where the
        # backward view groups by num_filters but forward indices range over
        # out_features.
        if param_shape is not None:
            out_dim = agg_bwd.shape[-1]
            original_shape = param_shape + (out_dim,)
            agg_bwd = mfx.swap_axes_and_reshape(
                mfx.invert_swap_axes_and_reshape(
                    agg_bwd, original_shape, bwd_index_dim, stride=bwd_stride
                ),
                fwd_index_dim,
                out_dim,
                stride=fwd_stride,
            )

        fwd_adjusted_inds_i, fwd_adjusted_inds_j = jnp.nonzero(
            param_mask_fwd_nv, size=num_cells_updated
        )

        fwd_attn_signals = agg_fwd[fwd_adjusted_inds_i, fwd_adjusted_inds_j, :]
        bwd_attn_signals = agg_bwd[fwd_adjusted_inds_i, fwd_adjusted_inds_j, :]

        focus_nv_fwd_concat = focus_vectors_fwd[fwd_adjusted_inds_i, fwd_adjusted_inds_j, :]

        perceptions = jnp.concatenate(
            [focus_nv_fwd_concat, fwd_attn_signals, bwd_attn_signals], axis=1
        )

        outputs = self.local_rule_head(perceptions)
        delta_theta = outputs[:, 0]
        delta_hidden = outputs[:, 1:]

        return delta_theta, delta_hidden, fwd_adjusted_inds_i, fwd_adjusted_inds_j
