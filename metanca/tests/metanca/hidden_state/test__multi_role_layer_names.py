from metanca.hidden_state import get_layer_names_from_shapes


def test_layer_names_include_embedding_and_scale():
    shapes = {
        "token_embed.embedding": (512, 32),
        "blocks_0.attn_norm.scale": (32,),
        "blocks_0.self_attn_qkv.kernel": (32, 96),
        "final_norm.scale": (32,),
        "lm_head.kernel": (32, 512),
    }
    names = get_layer_names_from_shapes(shapes)
    assert names == ["token_embed", "blocks_0.attn_norm", "blocks_0.self_attn_qkv",
                     "final_norm", "lm_head"]
