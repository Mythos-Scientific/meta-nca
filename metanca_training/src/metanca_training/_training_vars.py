from dataclasses import dataclass, field
from typing import Callable

import chex
import optax
import orbax.checkpoint as ocp

from metanca import TaskNet

from ._sample_pool import SamplePool


@dataclass(kw_only=True)
class TrainingVars:
    tasknets: list[TaskNet]
    test_tasknet: TaskNet

    rand_key: chex.PRNGKey
    hidden_dim: int

    local_rule_params: chex.ArrayTree
    local_rule_net_apply: Callable[..., chex.ArrayTree]
    optimizer: optax.GradientTransformation
    optimizer_state: optax.OptState
    update_step_scheduler: Callable[[int, int], int]
    n_update_steps: int

    checkpoint_manager: ocp.CheckpointManager
    checkpoint_metadata: dict
    start_metaepoch: int = 0

    # Sample pooling (one pool per training architecture, empty if disabled)
    pools: list[SamplePool] = field(default_factory=list)
