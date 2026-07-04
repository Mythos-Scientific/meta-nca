"""Helpers for computing configured metrics from batch scope."""

from collections.abc import Sequence

import chex
import jax
import jax.numpy as jnp

from ._registry import get_metric_spec


def _mean_metric_values(metric_dicts: Sequence[dict[str, chex.Numeric]]) -> dict[str, chex.Numeric]:
    if not metric_dicts:
        return {}

    metric_names = metric_dicts[0].keys()
    return {
        metric_name: jnp.mean(
            jnp.stack([jnp.asarray(metric_dict[metric_name]) for metric_dict in metric_dicts])
        )
        for metric_name in metric_names
    }


def _compute_logits_list(
    *,
    model_states: Sequence[chex.ArrayTree],
    apply_fns: Sequence,
    batch_x: jax.Array,
) -> list[jax.Array]:
    if len(model_states) != len(apply_fns):
        raise ValueError("model_states and apply_fns must have the same length")

    return [
        apply_fn(model_state, batch_x) for model_state, apply_fn in zip(model_states, apply_fns)
    ]


def compute_metrics(
    *,
    metric_names: Sequence[str],
    model_states: Sequence[chex.ArrayTree],
    apply_fns: Sequence,
    batch_x: jax.Array,
    batch_y: jax.Array,
    mask: jax.Array | None,
    loss: chex.Numeric | None = None,
    logits_list: Sequence[jax.Array] | None = None,
) -> dict[str, chex.Numeric]:
    """Compute the configured metrics for a batch."""
    if not metric_names:
        return {}

    specs = [get_metric_spec(metric_name) for metric_name in metric_names]
    needs_logits = any(spec.needs_logits for spec in specs)
    if needs_logits and logits_list is None:
        logits_list = _compute_logits_list(
            model_states=model_states,
            apply_fns=apply_fns,
            batch_x=batch_x,
        )

    metrics: dict[str, chex.Numeric] = {}
    for spec in specs:
        if spec.needs_logits:
            if logits_list is None:
                raise ValueError(f"{spec.name} requires logits")
            per_model_metrics = [
                spec.compute(logits=logits, labels=batch_y, mask=mask, loss=loss)
                for logits in logits_list
            ]
            metrics.update(_mean_metric_values(per_model_metrics))
            continue

        metrics.update(spec.compute(labels=batch_y, mask=mask, loss=loss))

    return metrics
