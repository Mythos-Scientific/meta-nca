"""Metric registry and metadata."""

from dataclasses import dataclass
from typing import Callable, Literal

import chex

from ._accuracy import compute_accuracy
from ._bpb import compute_bpb
from ._perplexity import compute_perplexity

MetricMode = Literal["min", "max"]
MetricFn = Callable[..., dict[str, chex.Numeric]]


@dataclass(frozen=True)
class MetricSpec:
    """Static metadata for a supported metric."""

    name: str
    compute: MetricFn
    mode: MetricMode
    needs_logits: bool = False
    needs_loss: bool = False


SUPPORTED_METRICS: dict[str, MetricSpec] = {
    "accuracy": MetricSpec(
        name="accuracy",
        compute=compute_accuracy,
        mode="max",
        needs_logits=True,
    ),
    "perplexity": MetricSpec(
        name="perplexity",
        compute=compute_perplexity,
        mode="min",
        needs_loss=True,
    ),
    "bpb": MetricSpec(
        name="bpb",
        compute=compute_bpb,
        mode="min",
        needs_loss=True,
    ),
}


def get_metric_spec(metric_name: str) -> MetricSpec:
    """Return the registered spec for a metric."""
    try:
        return SUPPORTED_METRICS[metric_name]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_METRICS))
        raise ValueError(
            f"Unsupported metric {metric_name!r}. Supported metrics: {supported}"
        ) from exc


def get_primary_metric_name(metric_names: list[str] | tuple[str, ...]) -> str:
    """Choose the primary configured metric, defaulting to loss."""
    return metric_names[0] if metric_names else "loss"


def get_metric_mode(metric_name: str) -> MetricMode:
    """Return whether lower or higher is better for the metric."""
    if metric_name == "loss":
        return "min"
    return get_metric_spec(metric_name).mode
