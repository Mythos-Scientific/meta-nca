import operator

import chex
import optax
import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_managers import preservation_policy, save_decision_policy


def create_checkpoint_manager(
    checkpoint_dir: str,
    save_top_n: int,
    monitor: str,
) -> ocp.CheckpointManager:
    if save_top_n <= 0:
        pres_policy = preservation_policy.PreserveAll()
    else:
        pres_policy = preservation_policy.BestN(
            n=save_top_n,
            get_metric_fn=operator.itemgetter(monitor),
        )
    checkpoint_options = ocp.CheckpointManagerOptions(
        save_decision_policy=save_decision_policy.FixedIntervalPolicy(1),
        preservation_policy=pres_policy,
    )
    return ocp.CheckpointManager(checkpoint_dir, options=checkpoint_options)


def build_checkpoint_metadata(optimizer_type: str, optimizer_kwargs: dict) -> dict:
    return {
        "optax_optimizer_type": optimizer_type,
        "optimizer_kwargs": optimizer_kwargs,
    }


def save_checkpoint(
    step: int,
    metrics: dict[str, float],
    params: chex.ArrayTree,
    opt_state: optax.OptState,
    metadata: dict,
    checkpoint_manager: ocp.CheckpointManager,
) -> None:
    checkpoint_manager.save(
        step,
        metrics=metrics,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave({"params": params, "opt_state": opt_state}),
            metadata=ocp.args.JsonSave(metadata | {"metaepoch": step}),
        ),
    )


def restore_checkpoint(
    step: int,
    abstract_state: chex.ArrayTree,
    checkpoint_manager: ocp.CheckpointManager,
) -> tuple[dict, chex.ArrayTree, optax.GradientTransformation, optax.OptState]:
    restored = checkpoint_manager.restore(
        step,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_state),
            metadata=ocp.args.JsonRestore(),
        ),
    )
    opt_state = restored["state"]["opt_state"]
    params = restored["state"]["params"]
    optimizer_type = restored["metadata"]["optax_optimizer_type"]
    optimizer_kwargs = restored["metadata"]["optimizer_kwargs"]
    checkpoint_metadata = build_checkpoint_metadata(optimizer_type, optimizer_kwargs)

    optimizer = operator.attrgetter(optimizer_type)(optax)(**optimizer_kwargs)
    return checkpoint_metadata, params, optimizer, opt_state
