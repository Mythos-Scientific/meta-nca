from importlib import util
from pathlib import Path


def _load_best_metrics_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "src" / "metanca_training" / "_best_metrics.py"
    )
    spec = util.spec_from_file_location("metanca_training_best_metrics", module_path)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metric_improvement_direction_uses_max_for_accuracy_and_min_for_loss() -> None:
    module = _load_best_metrics_module()

    assert module.METRIC_IMPROVEMENT_DIRECTION["accuracy"] is max
    assert module.METRIC_IMPROVEMENT_DIRECTION["loss"] is min


def test_update_best_metrics_tracks_best_values_by_metric_type() -> None:
    module = _load_best_metrics_module()

    best_metrics = module.update_best_metrics(
        {},
        {
            "val_arch/val_data_accuracy": 0.81,
            "val_arch/val_data_loss": 0.42,
            "train_archs/train_data_accuracy": 0.78,
            "train_archs/train_data_loss": 0.65,
            "train_archs/train_data_callback_time": 0.12,
        },
    )

    assert best_metrics == {
        "val_arch/val_data_accuracy": 0.81,
        "val_arch/val_data_loss": 0.42,
        "train_archs/train_data_accuracy": 0.78,
        "train_archs/train_data_loss": 0.65,
    }

    best_metrics = module.update_best_metrics(
        best_metrics,
        {
            "val_arch/val_data_accuracy": 0.79,
            "val_arch/val_data_loss": 0.51,
            "train_archs/train_data_accuracy": 0.82,
            "train_archs/train_data_loss": 0.59,
        },
    )

    assert best_metrics == {
        "val_arch/val_data_accuracy": 0.81,
        "val_arch/val_data_loss": 0.42,
        "train_archs/train_data_accuracy": 0.82,
        "train_archs/train_data_loss": 0.59,
    }


def test_build_metrics_with_best_adds_best_prefixed_metrics_for_tracked_metrics() -> None:
    module = _load_best_metrics_module()

    metrics = {
        "val_arch/val_data_accuracy": 0.81,
        "val_arch/val_data_loss": 0.42,
        "val_arch/val_data_callback_time": 0.09,
    }

    best_metrics = module.update_best_metrics({}, metrics)
    logged_metrics = module.build_metrics_with_best(metrics, best_metrics)

    assert logged_metrics == {
        "val_arch/val_data_accuracy": 0.81,
        "val_arch/val_data_loss": 0.42,
        "val_arch/val_data_callback_time": 0.09,
        "best_val_arch/val_data_accuracy": 0.81,
        "best_val_arch/val_data_loss": 0.42,
    }
