"""Hydra-based training entry point for MetaNCA.

Usage:
    python train.py dataset=mnist training.lr=0.001
    python train.py dataset=cifar100 checkpoint.run_name=my_experiment
    python train.py --config-name=config dataset=imagenet \
        dataset.imagenet_dir=/path/to/data \
        dataset.synset_mapping_file=/path/to/synsets.txt

    # Quick iteration mode
    python train.py training=fast wandb=disabled

    # Hyperparameter sweep
    python train.py -m dataset=mnist,cifar100 training.lr=0.001,0.0001

    # Architecture generalization evaluation
    python train.py dataset=cifar100_medium model=mnist_resnet \
        model.archs.0.num_classes=100 model.archs.1.num_classes=100 \
        model.test_arch.num_classes=100 model.test_arch.base_channels=20 \
        checkpoint.checkpoint_dir=local_rule_checkpoints/resnet_cifar100_pool_v2 \
        checkpoint.run_name=resnet_cifar100_pool_v2 \
        training.update_step_scheduler_max_steps=3 \
        +eval_arch_generalization=true \
        '+eval_base_channels=[8,12,14,16,18,20,22,24]' \
        +eval_n_samples=5
"""

import json
import logging
from pathlib import Path

import hydra
import jax
import matplotlib

matplotlib.use("Agg")

import numpy as np
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from metanca_training import train_metanca
from metanca_training._hydra_configs import register_configs
from metanca_training.data_utils import (
    get_fashion_mnist_datasets,
    get_iris_datasets,
    get_mnist_datasets,
    load_cifar100_arrays,
    load_imagenet_batched_sharded,
    prepare_batches,
)

# Register structured configs before Hydra initializes
register_configs()

logger = logging.getLogger(__name__)


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


def load_dataset(cfg: DictConfig, rand_key: jax.Array):
    """Load dataset based on configuration.

    Args:
        cfg: Hydra config with dataset settings
        rand_key: JAX random key for data shuffling

    Returns:
        Tuple of (train_batches_per_device, val_batches_per_device)
    """
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
        X = X.reshape(-1, *input_shape)
        train_batches, val_batches = prepare_batches(
            X, y, train_inds, val_inds, batch_size=batch_size
        )
    elif dataset_name == "fashion_mnist":
        X, y, train_inds, val_inds = get_fashion_mnist_datasets(rand_key)
        X = X.reshape(-1, *input_shape)
        train_batches, val_batches = prepare_batches(
            X, y, train_inds, val_inds, batch_size=batch_size
        )
    elif dataset_name == "cifar100":
        X, y, train_inds, val_inds = load_cifar100_arrays(rand_key)
        X = X.reshape(-1, *input_shape)
        train_batches, val_batches = prepare_batches(
            X, y, train_inds, val_inds, batch_size=batch_size
        )
    elif dataset_name == "imagenet":
        if not cfg.dataset.imagenet_dir or not cfg.dataset.synset_mapping_file:
            raise ValueError(
                "ImageNet requires dataset.imagenet_dir and "
                "dataset.synset_mapping_file to be set"
            )
        kwargs = dict(
            data_dir=cfg.dataset.imagenet_dir,
            synset_mapping_file=cfg.dataset.synset_mapping_file,
            rng_key=rand_key,
            batch_size=cfg.dataset.batch_size,
        )
        if cfg.dataset.get("max_per_class") is not None:
            kwargs["max_per_class"] = cfg.dataset.max_per_class
        train_batches, val_batches = load_imagenet_batched_sharded(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    logger.info(f"Done preparing batches {train_batches[0].shape}")
    return train_batches, val_batches


def setup_wandb(cfg: DictConfig) -> None:
    """Initialize wandb if enabled.

    Args:
        cfg: Hydra config with wandb settings
    """
    if cfg.wandb.enabled and not cfg.code_testing:
        wandb.init(
            name=cfg.checkpoint.run_name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            id=cfg.wandb.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        # Log architecture info
        wandb.config.update(
            {
                "train_archs": OmegaConf.to_container(cfg.model.archs),
                "test_arch": OmegaConf.to_container(cfg.model.test_arch),
            }
        )
    else:
        # Initialize wandb in disabled mode to avoid errors in training code
        wandb.init(mode="disabled")


def eval_arch_generalization(cfg: DictConfig) -> None:
    """Evaluate architecture generalization by sweeping base_channels.

    Uses the exact same setup as training (before_metanca_training) to ensure
    identical checkpoint loading and model construction.
    """
    import matplotlib.pyplot as plt
    import metanca
    from metanca.nn import ResNet

    from metanca_training._train_metanca import before_metanca_training
    from metanca_training._validation_step import metanca_validation_step
    from metanca_training.callbacks import CallbackRunner, create_accuracy_callback

    rand_key = jax.random.key(cfg.random.seed)

    models = [instantiate(arch_cfg) for arch_cfg in cfg.model.archs]
    test_model = instantiate(cfg.model.test_arch)

    training_vars = before_metanca_training(
        cfg,
        models=models,
        test_model=test_model,
        rand_key=rand_key,
    )

    _, val_batches = load_dataset(cfg, rand_key)

    input_shape = tuple(cfg.dataset.input_shape)
    n_spatial_dims = len(input_shape) - 1
    hidden_dim = training_vars.hidden_dim
    local_rule_params = training_vars.local_rule_params
    local_rule_net_apply = training_vars.local_rule_net_apply
    n_update_steps = training_vars.n_update_steps

    # Verify on the training test arch first
    print("=== Verifying on test arch ===", flush=True)
    cb = CallbackRunner.create([create_accuracy_callback()])
    metrics, _ = metanca_validation_step(
        val_batches=val_batches,
        local_rule_net_apply=local_rule_net_apply,
        local_rule_params=local_rule_params,
        rand_key=jax.random.key(42),
        val_tasknet=training_vars.test_tasknet,
        n_update_steps=n_update_steps,
        hidden_dim=hidden_dim,
        prop_cells_updated=cfg.training.prop_cells_updated,
        weight_transformer_dropout=cfg.training.weight_transformer_dropout,
        callback_runner=cb,
        n_spatial_dims=n_spatial_dims,
    )
    print(
        f"Test arch: accuracy={metrics.get('accuracy', 0):.4f}, "
        f"loss={metrics.get('loss', 0):.4f}",
        flush=True,
    )

    # Sweep config
    base_channels_list = list(cfg.get("eval_base_channels", [8, 12, 14, 16, 18, 20, 22, 24]))
    stage_blocks = list(cfg.model.test_arch.get("stage_blocks", [1, 1, 1]))
    n_samples = cfg.get("eval_n_samples", 5)
    output_shape = cfg.dataset.output_shape
    train_base_channels = {a.base_channels for a in cfg.model.archs}

    # Build eval TaskNets using the SAME shared initializer from training
    shared_init = (
        training_vars.test_tasknet.hidden_state_initializer,
        training_vars.test_tasknet.hidden_dim,
    )
    eval_models = [
        ResNet(
            num_classes=output_shape,
            stage_blocks=stage_blocks,
            base_channels=bc,
            use_bn=False,
            act="relu",
        )
        for bc in base_channels_list
    ]
    eval_keys = jax.random.split(jax.random.key(0), len(eval_models))
    eval_tasknets = [
        metanca.TaskNet.build(
            model=m,
            input_shape=input_shape,
            key=k,
            n_spatial_dims=n_spatial_dims,
            d_neuron=cfg.positional_encoding.d_neuron,
            d_spatial=cfg.positional_encoding.d_spatial,
            d_layer=cfg.positional_encoding.d_layer,
            shared_initializer=shared_init,
        )
        for m, k in zip(eval_models, eval_keys)
    ]

    results = {}
    for i, bc in enumerate(base_channels_list):
        is_train = bc in train_base_channels
        print(f"\nEvaluating base_channels={bc}{' (TRAIN)' if is_train else ''}", flush=True)

        accuracies = []
        for s in range(n_samples):
            cb = CallbackRunner.create([create_accuracy_callback()])
            metrics, _ = metanca_validation_step(
                val_batches=val_batches,
                local_rule_net_apply=local_rule_net_apply,
                local_rule_params=local_rule_params,
                rand_key=jax.random.key(s + 1000 * i),
                val_tasknet=eval_tasknets[i],
                n_update_steps=n_update_steps,
                hidden_dim=hidden_dim,
                prop_cells_updated=cfg.training.prop_cells_updated,
                weight_transformer_dropout=cfg.training.weight_transformer_dropout,
                callback_runner=cb,
                n_spatial_dims=n_spatial_dims,
            )
            acc = metrics.get("accuracy", 0.0)
            accuracies.append(acc)
            print(f"  Sample {s + 1}/{n_samples}: accuracy={acc:.4f}", flush=True)

        mean_acc = float(np.mean(accuracies))
        std_acc = float(np.std(accuracies))
        results[bc] = {"mean": mean_acc, "std": std_acc, "is_train": is_train}
        print(f"  => {mean_acc:.4f} +/- {std_acc:.4f}", flush=True)

    # Save and plot
    output_dir = Path("scripts/eval_results")
    output_dir.mkdir(exist_ok=True)
    run_name = cfg.checkpoint.run_name

    with open(output_dir / f"arch_gen_{run_name}.json", "w") as f:
        json.dump({"results": {str(k): v for k, v in results.items()}}, f, indent=2)

    bc_sorted = sorted(results.keys())
    means = [results[bc]["mean"] for bc in bc_sorted]
    stds = [results[bc]["std"] for bc in bc_sorted]
    is_train_list = [results[bc]["is_train"] for bc in bc_sorted]

    fig, ax = plt.subplots(figsize=(max(8, len(bc_sorted) * 1.2), 3))
    acc_array = np.array(means).reshape(1, -1)
    vmax = max(means) * 1.1 if max(means) > 0 else 1.0
    im = ax.imshow(acc_array, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(bc_sorted)))
    ax.set_xticklabels([str(bc) for bc in bc_sorted])
    ax.set_xlabel("base_channels")
    ax.set_yticks([0])
    ax.set_yticklabels([f"blocks={stage_blocks}"])
    for j, bc in enumerate(bc_sorted):
        color = "white" if means[j] > 0.5 * vmax else "black"
        ax.text(
            j,
            0,
            f"{means[j]:.3f}\n+/-{stds[j]:.3f}",
            ha="center",
            va="center",
            color=color,
            fontsize=9,
        )
        if is_train_list[j]:
            ax.add_patch(
                plt.Rectangle((j - 0.5, -0.5), 1, 1, linewidth=3, edgecolor="red", facecolor="none")
            )
    plt.colorbar(im, ax=ax, label="Mean Accuracy")
    ax.set_title(
        f"Arch Gen: {cfg.dataset.name.upper()} ResNet - {run_name}\n"
        f"({n_samples} samples, {n_update_steps} steps)"
    )
    plt.tight_layout()

    plot_path = str(output_dir / f"arch_gen_{run_name}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {plot_path}", flush=True)
    plt.close()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration.

    Args:
        cfg: Hydra DictConfig populated from YAML files and CLI overrides
    """
    setup_logging()

    # Log resolved configuration
    for line in OmegaConf.to_yaml(cfg).splitlines():
        logger.info(line)

    # Check for eval mode
    if cfg.get("eval_arch_generalization", False):
        wandb.init(mode="disabled")
        eval_arch_generalization(cfg)
        return

    # Setup wandb
    setup_wandb(cfg)

    # Create JAX random key
    rand_key = jax.random.key(cfg.random.seed)

    # Load dataset
    train_batches, val_batches = load_dataset(cfg, rand_key)

    models = [instantiate(arch_cfg) for arch_cfg in cfg.model.archs]
    test_model = instantiate(cfg.model.test_arch)

    # Run training
    train_metanca(train_batches, val_batches, cfg=cfg, models=models, test_model=test_model)


if __name__ == "__main__":
    main()
