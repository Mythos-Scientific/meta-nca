"""Hydra-based entry point for regular net (SGD/backprop) baseline training.

Usage:
    python train_regular_net.py dataset=mnist training.num_epochs=10000
    python train_regular_net.py dataset=cifar100 training.num_epochs=2000
    python train_regular_net.py wandb=disabled
"""

import logging
import time

import hydra
import jax
import optax
import wandb
from etils import epath
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from metanca_training._checkpointing import (
    build_checkpoint_metadata,
    create_checkpoint_manager,
    restore_checkpoint,
    save_checkpoint,
)
from metanca_training._hydra_configs import register_configs
from metanca_training.data_utils import (
    get_iris_datasets,
    get_mnist_datasets,
    load_cifar100_arrays,
    load_imagenet_batched_sharded,
    prepare_batches,
)
from metanca_training.regular_net_functions import (
    evaluate_reg_net_batched,
    init_and_train_regular_network_batched,
)

# Register structured configs before Hydra initializes
register_configs()

logger = logging.getLogger(__name__)


def _to_expected_input_layout(X, input_shape: tuple[int, ...]):
    """Convert NCHW arrays to NHWC when config expects NHWC."""
    if X.ndim == 4 and len(input_shape) == 3:
        # Typical image case: source is NCHW from loaders, config is NHWC.
        if X.shape[1] == input_shape[-1] and X.shape[-1] != input_shape[-1]:
            return X.transpose(0, 2, 3, 1)
    return X.reshape(-1, *input_shape)


def setup_logging() -> None:
    """Configure logging with sensible defaults."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    # Suppress noisy loggers
    for lib in ["jax", "jaxlib", "matplotlib", "absl", "asyncio"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def setup_wandb(cfg: DictConfig) -> None:
    """Initialize wandb if enabled."""
    if cfg.wandb.enabled and not cfg.code_testing:
        wandb.init(
            name=cfg.checkpoint.run_name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update(
            {
                "train_archs": OmegaConf.to_container(cfg.model.archs),
                "test_arch": OmegaConf.to_container(cfg.model.test_arch),
            }
        )
    else:
        wandb.init(mode="disabled")


def load_dataset(cfg: DictConfig, rand_key: jax.Array):
    """Load dataset based on configuration."""
    dataset_name = cfg.dataset.name
    batch_size = cfg.dataset.batch_size if cfg.dataset.batch_size > 0 else "all"
    input_shape = tuple(cfg.dataset.input_shape)

    logger.info(f"Preparing [{dataset_name}] batches")

    if dataset_name == "iris":
        X, y, train_inds, val_inds = get_iris_datasets(rand_key)
        train_batches, val_batches = prepare_batches(
            X, y, train_inds, val_inds, batch_size=batch_size
        )
    elif dataset_name == "mnist":
        X, y, train_inds, val_inds = get_mnist_datasets(rand_key, reshape=True)
        X = _to_expected_input_layout(X, input_shape)
        train_batches, val_batches = prepare_batches(
            X, y, train_inds, val_inds, batch_size=batch_size
        )
    elif dataset_name == "cifar100":
        X, y, train_inds, val_inds = load_cifar100_arrays(rand_key)
        X = _to_expected_input_layout(X, input_shape)
        train_batches, val_batches = prepare_batches(
            X, y, train_inds, val_inds, batch_size=batch_size
        )
    elif dataset_name == "imagenet":
        if not cfg.dataset.imagenet_dir or not cfg.dataset.synset_mapping_file:
            raise ValueError(
                "ImageNet requires dataset.imagenet_dir and "
                "dataset.synset_mapping_file to be set"
            )
        train_batches, val_batches = load_imagenet_batched_sharded(
            cfg.dataset.imagenet_dir,
            cfg.dataset.synset_mapping_file,
            rand_key,
            batch_size=cfg.dataset.batch_size,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    logger.info(f"Done preparing batches {train_batches[0].shape}")
    return train_batches, val_batches


def _resolve_regular_net_models(cfg: DictConfig) -> tuple[list[object], list[str]]:
    selection = str(getattr(cfg.training, "regular_net_arch_selection", "train")).strip().lower()
    if not selection:
        selection = (
            "both" if bool(getattr(cfg.training, "train_test_arch_regular_net", False)) else "train"
        )
    if selection not in {"train", "test", "both"}:
        raise ValueError(
            "training.regular_net_arch_selection must be one of: train, test, both "
            f"(got: {selection})"
        )

    models: list[object] = []
    labels: list[str] = []
    if selection in {"train", "both"}:
        train_models = [instantiate(arch_cfg) for arch_cfg in cfg.model.archs]
        models.extend(train_models)
        labels.extend([f"train_{i}" for i in range(len(train_models))])
    if selection in {"test", "both"}:
        models.append(instantiate(cfg.model.test_arch))
        labels.append("test")
    return models, labels


@hydra.main(version_base=None, config_path="../configs", config_name="config_regular_net")
def main(cfg: DictConfig) -> None:
    """Run regular net baseline training with Hydra configuration."""
    setup_logging()

    # Log resolved configuration
    for line in OmegaConf.to_yaml(cfg).splitlines():
        logger.info(line)

    setup_wandb(cfg)

    rand_key = jax.random.key(cfg.random.seed)
    train_batches, val_batches = load_dataset(cfg, rand_key)

    models, arch_labels = _resolve_regular_net_models(cfg)

    logger.info(f"Regular net architectures ({len(models)}): {arch_labels}")
    epochs = int(cfg.training.num_epochs)

    for arch_idx, (reg_net_model, arch_label) in enumerate(zip(models, arch_labels)):
        logger.info(f"Training regular net arch {arch_idx} [{arch_label}] for {epochs} epochs")
        start = time.time()
        checkpoint_dir = (
            epath.Path(cfg.checkpoint.checkpoint_dir) / f"regular_net_{arch_label}"
        ).resolve()
        checkpoint_manager = create_checkpoint_manager(
            checkpoint_dir,
            cfg.checkpoint.save_top_n,
            monitor="val_accuracy",
        )

        optimizer_type = "adam"
        optimizer_kwargs = {"learning_rate": 0.001}
        optimizer = optax.adam(**optimizer_kwargs)
        checkpoint_metadata = build_checkpoint_metadata(optimizer_type, optimizer_kwargs)

        (
            rand_key,
            reg_net_model_state,
            _reg_net_params_history,
            model,
            optimizer,
            opt_state,
        ) = init_and_train_regular_network_batched(
            rand_key,
            None,
            train_batches,
            val_batches,
            epochs=0,
            model=reg_net_model,
            optimizer=optimizer,
        )

        start_epoch = 0
        if (step := checkpoint_manager.latest_step()) is not None:
            logger.info(f"Restoring regular net checkpoint (step {step})")
            checkpoint_metadata, reg_net_model_state, optimizer, opt_state = restore_checkpoint(
                step,
                {"params": reg_net_model_state, "opt_state": opt_state},
                checkpoint_manager,
            )
            start_epoch = step

        epochs_to_train = max(0, epochs - start_epoch)
        if epochs_to_train > 0:
            (
                rand_key,
                reg_net_model_state,
                _reg_net_params_history,
                model,
                optimizer,
                opt_state,
            ) = init_and_train_regular_network_batched(
                rand_key,
                None,
                train_batches,
                val_batches,
                epochs=epochs_to_train,
                params=reg_net_model_state,
                optimizer=optimizer,
                opt_state=opt_state,
                model=reg_net_model,
            )

        train_acc = evaluate_reg_net_batched(model, reg_net_model_state, train_batches)
        val_acc = evaluate_reg_net_batched(model, reg_net_model_state, val_batches)

        save_checkpoint(
            start_epoch + epochs_to_train,
            {"val_accuracy": float(val_acc), "train_accuracy": float(train_acc)},
            reg_net_model_state,
            opt_state,
            checkpoint_metadata,
            checkpoint_manager,
        )
        checkpoint_manager.wait_until_finished()

        duration = time.time() - start
        logger.info(
            "Regular net results | arch=%s label=%s train_acc=%.4f val_acc=%.4f time=%.2fs",
            arch_idx,
            arch_label,
            float(train_acc),
            float(val_acc),
            duration,
        )
        if cfg.wandb.enabled and not cfg.code_testing:
            wandb.log(
                {
                    "regular_net/arch_index": arch_idx,
                    "regular_net/arch_label": arch_label,
                    "regular_net/train_accuracy": float(train_acc),
                    "regular_net/val_accuracy": float(val_acc),
                    "regular_net/train_time_seconds": duration,
                }
            )


if __name__ == "__main__":
    main()
