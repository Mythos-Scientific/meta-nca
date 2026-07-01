import jax
import wandb
from hydra import compose, initialize_config_dir
from pathlib import Path

from metanca_training._hydra_configs import register_configs
from metanca_training import train_metanca
from metanca_training.data_utils import get_iris_datasets, prepare_batches
from metanca_training.scaling.arch_grid import build_mlp

CONFIG_DIR = str((Path(__file__).parents[2] / "configs").resolve())


def test_train_metanca_returns_params_and_vars():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=iris", "wandb=disabled",
                "training.num_metaepochs=2",
                "training.update_step_scheduler_max_steps=1",
                "training.early_stopping_enabled=false",
                "checkpoint.run_name=pytest_smoke",
                "logging.log_every_n_metaepochs=1",
            ],
        )
    key = jax.random.key(0)
    X, y, tr, va = get_iris_datasets(key)
    train_b, val_b = prepare_batches(X, y, tr, va, batch_size="all")
    wandb.init(mode="disabled")
    models = [build_mlp((4,), n_classes=3)]  # tiny MLP on iris (4 features, 3 classes)
    params, tvars = train_metanca(
        train_b, val_b, cfg=cfg, models=models, test_model=build_mlp((4,), n_classes=3)
    )
    assert params is not None
    assert hasattr(tvars, "test_tasknet") and hasattr(tvars, "local_rule_net_apply")
