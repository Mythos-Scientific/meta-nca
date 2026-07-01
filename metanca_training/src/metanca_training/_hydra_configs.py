"""Hydra structured configs for metanca_training.

These dataclasses serve as schemas for YAML configs and provide type validation.
Training code consumes this Hydra configuration directly.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ConvLayerConfig:
    """Single convolutional layer specification."""

    num_channels: int = MISSING
    kernel_size: list[int] = MISSING
    pool_size: list[int] = MISSING
    pool_strides: list[int] = MISSING


@dataclass
class ArchitectureConfig:
    """TaskNet architecture specification."""

    _target_: Optional[str] = None

    # MLP
    layer_specs: list[int] = field(default_factory=list)
    activation: str = "leaky_relu"
    final_layer_activation: str = "identity"
    use_bias: Optional[bool] = None

    # ConvMLP
    conv_layer_specs: list[Any] = field(default_factory=list)
    mlp_layer_specs: list[int] = field(default_factory=list)
    conv_use_bias: Optional[bool] = None
    mlp_use_bias: Optional[bool] = None
    padding: str = "VALID"
    strides: int = 1

    # ResNet
    num_classes: Optional[int] = None
    stage_blocks: Optional[list[int]] = None
    base_channels: int = 16
    use_bn: bool = False
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    act: str = "relu"


@dataclass
class DatasetConfig:
    """Dataset-specific configuration."""

    name: str = MISSING
    input_shape: list[int] = MISSING
    output_shape: int = MISSING
    batch_size: int = 0  # 0 means use entire dataset

    # ImageNet-specific (optional)
    imagenet_dir: Optional[str] = None
    synset_mapping_file: Optional[str] = None
    max_per_class: Optional[int] = None


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    use_bias: bool = True
    archs: list[ArchitectureConfig] = MISSING
    test_arch: ArchitectureConfig = MISSING


@dataclass
class PositionalEncodingConfig:
    """Positional encoding dimensions."""

    d_layer: int = 10
    d_neuron: int = 10
    d_spatial: int = 10


@dataclass
class LocalRuleConfig:
    """Local rule network configuration.

    Only hidden layer sizes are specified here - input and output dimensions
    are computed at runtime based on hidden_state_dim.
    """

    hidden_layers: list[int] = field(default_factory=lambda: [100, 200, 100])


@dataclass
class SamplePoolingConfig:
    """Sample pooling configuration (Growing NCA, Mordvintsev et al. 2020)."""

    enabled: bool = False
    pool_size: int = 64
    seed_fraction: float = 0.2
    loss_spike_multiplier: float = 10.0
    max_consecutive_rollbacks: int = 3


@dataclass
class MetaNCATrainingHyperparamsConfig:
    """MetaNCA training hyperparameters."""

    lr: float = 1e-3
    num_epochs: int = 1
    num_metaepochs: int = 100000
    grad_clip_norm: float = 1.0
    weight_transformer_dropout: float = 0.0
    prop_cells_updated: float = 0.8

    # Update step scheduler
    update_step_scheduler_type: str = "increment"
    update_step_scheduler_rate: int = 100
    update_step_scheduler_max_steps: Optional[int] = None

    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 10

    # Sample pooling
    sample_pooling: SamplePoolingConfig = field(default_factory=SamplePoolingConfig)


@dataclass
class RegularNetTrainingHyperparamsConfig:
    """Regular-net baseline training hyperparameters."""

    num_epochs: int = 1
    # one of: train, test, both
    regular_net_arch_selection: str = "train"
    # Legacy compatibility fallback in train_regular_net.py
    train_test_arch_regular_net: bool = False


@dataclass
class LoggingConfig:
    """MetaNCA logging configuration."""

    log_every_n_metaepochs: int = 10
    log_every_n_batches: int = 25


@dataclass
class MetaNCACheckpointConfig:
    """MetaNCA checkpointing configuration."""

    checkpoint_dir: str = "local_rule_checkpoints"
    run_name: str = "no_name"
    save_top_n: int = 0
    load_local_rule_net: Optional[str] = None


@dataclass
class RegularNetCheckpointConfig:
    """Regular-net checkpointing configuration."""

    checkpoint_dir: str = "regular_net_checkpoints"
    run_name: str = "no_name"
    save_top_n: int = 0


@dataclass
class WandBConfig:
    """Weights & Biases configuration."""

    enabled: bool = True
    project: str = "metanca"
    entity: Optional[str] = None
    run_id: Optional[str] = None


@dataclass
class RandomConfig:
    """Random seed configuration."""

    seed: int = 0


@dataclass
class MetaNCAConfig:
    """Root configuration for MetaNCA training."""

    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING
    positional_encoding: PositionalEncodingConfig = field(default_factory=PositionalEncodingConfig)
    local_rule: LocalRuleConfig = field(default_factory=LocalRuleConfig)
    training: MetaNCATrainingHyperparamsConfig = field(
        default_factory=MetaNCATrainingHyperparamsConfig
    )
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: MetaNCACheckpointConfig = field(default_factory=MetaNCACheckpointConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    random: RandomConfig = field(default_factory=RandomConfig)

    # Legacy flags (can be deprecated over time)
    code_testing: bool = False
    linearly_separable: bool = False
    add_pos_enc: bool = False
    zero_out: float = 0.0


@dataclass
class RegularNetConfig:
    """Root configuration for regular-net baseline training."""

    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING
    training: RegularNetTrainingHyperparamsConfig = field(
        default_factory=RegularNetTrainingHyperparamsConfig
    )
    checkpoint: RegularNetCheckpointConfig = field(default_factory=RegularNetCheckpointConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    random: RandomConfig = field(default_factory=RandomConfig)

    code_testing: bool = False


def register_configs() -> None:
    """Register all structured configs with Hydra's ConfigStore."""
    cs = ConfigStore.instance()

    # Register root schemas
    cs.store(name="config_schema", node=MetaNCAConfig)
    cs.store(name="config_regular_net_schema", node=RegularNetConfig)

    # Register group schemas for validation
    cs.store(group="dataset", name="dataset_schema", node=DatasetConfig)
    cs.store(group="model", name="model_schema", node=ModelConfig)
    cs.store(group="training", name="metanca_schema", node=MetaNCATrainingHyperparamsConfig)
    cs.store(group="training", name="regular_net_schema", node=RegularNetTrainingHyperparamsConfig)
    cs.store(group="checkpoint", name="metanca_schema", node=MetaNCACheckpointConfig)
    cs.store(group="checkpoint", name="regular_net_schema", node=RegularNetCheckpointConfig)
