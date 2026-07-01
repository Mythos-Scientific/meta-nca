# metanca_training/tests/metanca_training/scaling/test__evaluate_pool.py
import jax
from pathlib import Path
from hydra import compose, initialize_config_dir

from metanca_training._hydra_configs import register_configs
from metanca_training._train_metanca import before_metanca_training
from metanca_training.data_utils import get_fashion_mnist_datasets, prepare_batches
from metanca_training.scaling.arch_grid import build_mlp
from metanca_training.scaling.hidden_state import grid_hidden_state_initializer
from metanca_training.scaling.evaluate_pool import evaluate_arch_pool

CONFIG_DIR = str((Path(__file__).parents[3] / "configs").resolve())  # metanca_training/configs


def test_evaluate_arch_pool_rows():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config",
                      overrides=["dataset=fashion_mnist", "wandb=disabled",
                                 "checkpoint.run_name=pytest_eval"])
    key = jax.random.key(0)
    X, y, tr, va = get_fashion_mnist_datasets(key)
    _, val_b = prepare_batches(X, y, tr, va, batch_size=512)
    # explicit grid-max initializer (neuron 784, layer 6) covers the depth-2 eval arch below
    shared = grid_hidden_state_initializer(784, d_neuron=10, d_layer=10, d_spatial=10)
    tvars = before_metanca_training(
        cfg, models=[build_mlp((32,), 10)], test_model=build_mlp((64, 32), 10),
        rand_key=key, shared_initializer=shared,
    )
    rows = evaluate_arch_pool(
        training_vars=tvars, local_rule_params=tvars.local_rule_params,
        archs=[((32,), "train"), ((64, 32), "val")],
        val_batches=val_b, cfg=cfg, n_update_steps=1, n_init_samples=2, rand_key=key,
    )
    assert len(rows) == 2
    keys = {"arch_id", "depth", "split", "val_loss_mean", "val_loss_std",
            "val_acc_mean", "val_acc_std"}
    for r in rows:
        assert keys <= set(r)
        assert r["val_loss_mean"] == r["val_loss_mean"]  # not NaN

    # resume + incremental-write behavior
    seen: list[dict] = []
    rows2 = evaluate_arch_pool(
        training_vars=tvars, local_rule_params=tvars.local_rule_params,
        archs=[((32,), "train"), ((64, 32), "val")],
        val_batches=val_b, cfg=cfg, n_update_steps=1, n_init_samples=1, rand_key=key,
        skip_ids={"d1_32"}, on_row=seen.append,
    )
    assert [r["arch_id"] for r in rows2] == ["d2_64-32"]   # skipped d1_32
    assert seen == rows2                                    # on_row fired per surviving arch
