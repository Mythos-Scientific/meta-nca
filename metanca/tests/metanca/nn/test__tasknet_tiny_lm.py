import jax
import jax.numpy as jnp

import metanca
from metanca.nn import TinyCausalLM


def _lm(vocab=512, d=32, heads=2):
    return TinyCausalLM(vocab_size=vocab, d_model=d, num_heads=heads,
                        num_layers=3, mlp_dim=4 * d, max_seq_len=64)


def test_tasknet_builds_for_tiny_lm():
    tn = metanca.TaskNet.build(
        model=_lm(), input_shape=(64,), key=jax.random.key(0),
        n_spatial_dims=0, d_neuron=10, d_layer=10, d_spatial=10,
        dummy_input_dtype=jnp.int32,
    )
    names = list(tn.iter_param_names())
    assert any(n.endswith(".embedding") for n in names)
    assert any(n.endswith(".scale") for n in names)
    assert any("self_attn_qkv" in n for n in names)
    # every param has a hidden state of matching leading shape
    for n in names:
        p = tn.get(n, "param"); h = tn.get(n, "hidden_state")
        assert h.shape[:-1] == p.shape, n
    # reset must also work (uses the same dtype/multi-role paths)
    tn2 = tn.reset(jax.random.key(1))
    assert set(tn2.iter_param_names()) == set(names)


def test_build_many_shared_initializer_still_works_for_mlp():
    # regression: the public shared_initializer path must survive the merge
    from metanca.nn import MultiLayerPerceptron
    m = MultiLayerPerceptron(layer_specs=[16, 10])
    tns = metanca.TaskNet.build_many(models=[m], input_shapes=[(784,)],
                                     key=jax.random.key(0), d_neuron=10,
                                     d_layer=10, d_spatial=10)
    assert tns[0].hidden_dim == 30
