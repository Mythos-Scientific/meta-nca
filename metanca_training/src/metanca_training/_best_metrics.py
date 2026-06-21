from collections.abc import Mapping

METRIC_IMPROVEMENT_DIRECTION = {
    "accuracy": max,
    "loss": min,
}


def _metric_type(metric_name: str) -> str | None:
    for metric_type in METRIC_IMPROVEMENT_DIRECTION:
        if metric_name == metric_type or metric_name.endswith(f"_{metric_type}"):
            return metric_type
    return None


def update_best_metrics(
    best_metrics: Mapping[str, float], metrics: Mapping[str, float]
) -> dict[str, float]:
    updated_best_metrics = dict(best_metrics)

    for metric_name, metric_value in metrics.items():
        metric_type = _metric_type(metric_name)
        if metric_type is None:
            continue

        current_value = float(metric_value)
        if metric_name in updated_best_metrics:
            updated_best_metrics[metric_name] = METRIC_IMPROVEMENT_DIRECTION[metric_type](
                updated_best_metrics[metric_name], current_value
            )
        else:
            updated_best_metrics[metric_name] = current_value

    return updated_best_metrics


def build_metrics_with_best(
    metrics: Mapping[str, float], best_metrics: Mapping[str, float]
) -> dict[str, float]:
    logged_metrics = dict(metrics)

    for metric_name in metrics:
        if metric_name in best_metrics and _metric_type(metric_name) is not None:
            logged_metrics[f"best_{metric_name}"] = best_metrics[metric_name]

    return logged_metrics
