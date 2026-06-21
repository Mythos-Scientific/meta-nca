import jax
import jax.numpy as jnp

from metanca_training._loss import masked_softmax_cross_entropy
from metanca_training.regular_net_functions import init_and_train_regular_network_batched


def test_masked_softmax_cross_entropy_all_false_mask_returns_zero() -> None:
    """All-false mask should be treated as an empty slice with zero contribution."""
    logits = jnp.array([[1.0, -1.0]], dtype=jnp.float32)
    labels = jnp.array([[1.0, 0.0]], dtype=jnp.float32)
    mask = jnp.array([[False]], dtype=bool)

    loss = masked_softmax_cross_entropy(logits, labels, mask)

    assert jnp.isfinite(loss)
    assert float(loss) == 0.0


def test_regular_net_batched_training_stays_finite_with_padded_false_mask(capsys) -> None:
    """Padded masked elements should not produce NaN training/validation loss."""
    key = jax.random.key(0)

    # Shape follows prepare_batches output: (n_batches, batch_size, ...)
    train_x = jnp.array([[[0.1, -0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0]]], dtype=jnp.float32)
    train_y = jnp.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=jnp.float32)
    train_mask = jnp.array([[[True], [False]]], dtype=bool)

    val_x = train_x
    val_y = train_y
    val_mask = train_mask

    arch = ([], [4, 2])

    init_and_train_regular_network_batched(
        key,
        arch,
        (train_x, train_y, train_mask),
        (val_x, val_y, val_mask),
        epochs=1,
    )

    out = capsys.readouterr().out.lower()
    assert "train loss at step 0: nan" not in out
    assert "val loss: nan" not in out
