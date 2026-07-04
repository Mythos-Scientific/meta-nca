# metanca_training/tests/metanca_training/scaling/test__plot_scaling.py
import json
from pathlib import Path

from metanca_training.scaling.plotting import aggregate, load_results


def _write(tmp_path):
    rows = [
        {"T": 1, "rep": 0, "split": "train", "val_loss_mean": 0.5, "val_acc_mean": 0.8},
        {"T": 1, "rep": 0, "split": "val", "val_loss_mean": 0.9, "val_acc_mean": 0.6},
        {"T": 5, "rep": 0, "split": "train", "val_loss_mean": 0.4, "val_acc_mean": 0.85},
        {"T": 5, "rep": 0, "split": "val", "val_loss_mean": 0.7, "val_acc_mean": 0.7},
    ]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_load_and_aggregate(tmp_path):
    rows = load_results(_write(tmp_path))
    agg = aggregate(rows, "loss")
    assert set(agg) == {1, 5}
    assert agg[1]["val"] == [0.9]
    assert agg[5]["train"] == [0.4]


def test_plot_writes_png(tmp_path):
    from metanca_training.scaling.plotting import plot_boxplots
    agg = aggregate(load_results(_write(tmp_path)), "acc")
    out = tmp_path / "box.png"
    plot_boxplots(agg, "acc", out)
    assert out.exists() and out.stat().st_size > 0


def _write_bpb(tmp_path):
    rows = [
        {"T": 1, "rep": 0, "split": "train", "val_bpb_mean": 1.2},
        {"T": 1, "rep": 0, "split": "val", "val_bpb_mean": 1.5},
        {"T": 5, "rep": 0, "split": "train", "val_bpb_mean": 1.1},
        {"T": 5, "rep": 0, "split": "val", "val_bpb_mean": 1.3},
    ]
    p = tmp_path / "bpb.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_aggregate_bpb(tmp_path):
    rows = load_results(_write_bpb(tmp_path))
    agg = aggregate(rows, "bpb")
    assert set(agg) == {1, 5}
    assert agg[1]["val"] == [1.5]
    assert agg[5]["train"] == [1.1]


def test_plot_boxplots_bpb_with_adam(tmp_path):
    from metanca_training.scaling.plotting import plot_boxplots
    rows = load_results(_write_bpb(tmp_path))
    agg = aggregate(rows, "bpb")
    out = tmp_path / "b.png"
    adam = {"llm_d32_h2_v512": {"val_loss": 1.0, "val_ppl": 2.7, "val_bpb": 1.4}}
    plot_boxplots(agg, "bpb", out, adam=adam)
    assert out.exists() and out.stat().st_size > 0
