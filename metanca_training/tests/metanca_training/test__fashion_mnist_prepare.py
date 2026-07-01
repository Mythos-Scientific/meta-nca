import jax
import numpy as np

from metanca_training.data_utils import _prepare_fashion_mnist_arrays


def test_prepare_fashion_mnist_shapes_and_split():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(100, 28, 28), dtype=np.uint8)
    labels = rng.integers(0, 10, size=(100,), dtype=np.int32)

    X, y, train_inds, val_inds = _prepare_fashion_mnist_arrays(
        images, labels, jax.random.key(0)
    )

    assert X.shape == (100, 784) and X.dtype == np.float32
    assert y.shape == (100, 10)
    assert len(train_inds) == 80 and len(val_inds) == 20
    assert set(np.array(train_inds)).isdisjoint(set(np.array(val_inds)))
    # standardized per feature: near-zero mean
    assert abs(float(X.mean())) < 1e-3
