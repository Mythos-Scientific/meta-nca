from importlib import util
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from metanca_training.regular_net_functions import init_and_train_regular_network_batched


class Rank2OnlyModel(nn.Module):
    """Fails if training loop feeds per-example rank-1 vectors."""

    @nn.compact
    def __call__(self, x: jax.Array, *, training: bool = False) -> jax.Array:
        if x.ndim != 2:
            raise ValueError(f"Expected batched rank-2 inputs, got shape={x.shape}")
        x = nn.Dense(features=8)(x)
        x = nn.relu(x)
        return nn.Dense(features=2)(x)


def _load_train_regular_net_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "train_regular_net.py"
    spec = util.spec_from_file_location("train_regular_net_script", script_path)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batched_regular_net_training_uses_batched_inputs() -> None:
    key = jax.random.key(0)

    # (n_batches, batch_size, features)
    train_x = jnp.array(
        [
            [[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.2, 0.3], [0.5, 0.4, 0.2, 0.1]],
            [[0.7, 0.1, 0.2, 0.5], [0.1, 0.6, 0.2, 0.1], [0.2, 0.3, 0.8, 0.4]],
        ],
        dtype=jnp.float32,
    )
    train_y = jnp.array(
        [
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=jnp.float32,
    )
    train_m = jnp.ones((2, 3, 1), dtype=bool)

    val_x, val_y, val_m = train_x, train_y, train_m

    _, model_state, _, _, _, _ = init_and_train_regular_network_batched(
        key,
        layer_widths=([], [4, 2]),
        train_batches_per_device=(train_x, train_y, train_m),
        val_batches_per_device=(val_x, val_y, val_m),
        epochs=1,
        model=Rank2OnlyModel(),
    )

    leaves = jax.tree_util.tree_leaves(model_state)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in leaves)


def test_to_expected_input_layout_transposes_nchw_to_nhwc() -> None:
    module = _load_train_regular_net_module()

    x = np.arange(1 * 3 * 2 * 2, dtype=np.float32).reshape(1, 3, 2, 2)
    out = module._to_expected_input_layout(x, (2, 2, 3))

    expected = np.transpose(x, (0, 2, 3, 1))
    assert out.shape == (1, 2, 2, 3)
    assert np.array_equal(out, expected)
