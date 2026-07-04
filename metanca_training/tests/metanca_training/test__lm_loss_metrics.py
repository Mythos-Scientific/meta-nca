import numpy as np
import jax.numpy as jnp
from metanca_training._loss import compute_loss, masked_sparse_softmax_cross_entropy
from metanca_training.metrics import compute_metrics


def test_sparse_ce_matches_manual():
    logits = jnp.zeros((2, 3, 5))                      # uniform -> CE = ln(5)
    targets = jnp.array([[1, 2, 3], [0, 0, 0]], dtype=jnp.int32)
    mask = jnp.ones((2, 3, 1), dtype=bool)
    loss = masked_sparse_softmax_cross_entropy(logits, targets, mask)
    assert np.isclose(float(loss), np.log(5), atol=1e-5)


def test_compute_loss_dispatches_sparse():
    apply_fn = lambda p, x: jnp.zeros((x.shape[0], x.shape[1], 7))
    x = jnp.zeros((2, 4), dtype=jnp.int32)
    y = jnp.zeros((2, 4), dtype=jnp.int32)
    m = jnp.ones((2, 4, 1), dtype=bool)
    assert np.isclose(float(compute_loss(x, y, m, {}, apply_fn)), np.log(7), atol=1e-5)


def test_bpb_and_perplexity():
    # metrics/__init__.py exports compute_metrics (not bare bpb/perplexity fns);
    # exercise both metrics through that public entry point.
    bpb_metrics = compute_metrics(
        metric_names=["bpb"],
        model_states=[],
        apply_fns=[],
        batch_x=None,
        batch_y=None,
        mask=None,
        loss=jnp.log(2.0),
    )
    assert np.isclose(float(bpb_metrics["bpb"]), 1.0)

    ppl_metrics = compute_metrics(
        metric_names=["perplexity"],
        model_states=[],
        apply_fns=[],
        batch_x=None,
        batch_y=None,
        mask=None,
        loss=jnp.array(0.0),
    )
    assert np.isclose(float(ppl_metrics["perplexity"]), 1.0)
