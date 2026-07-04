# metanca_training/src/metanca_training/scaling/plotting.py
"""Load scaling results and render boxplot + scatter figures per metric."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_MEAN_KEY = {
    "loss": "val_loss_mean",
    "acc": "val_acc_mean",
    "bpb": "val_bpb_mean",
    "ppl": "val_ppl_mean",
}
_ADAM_KEY = {
    "loss": "val_loss",
    "acc": "val_acc",
    "bpb": "val_bpb",
    "ppl": "val_ppl",
}
_LABEL = {
    "loss": "validation loss",
    "acc": "validation accuracy",
    "bpb": "validation bits/byte",
    "ppl": "validation perplexity",
}


def load_results(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_results_dir(dirpath) -> list[dict]:
    """Concatenate all per-run *.jsonl files under a results dir (e.g. results/scaling/fixed5)."""
    rows: list[dict] = []
    for f in sorted(Path(dirpath).glob("*.jsonl")):
        rows.extend(load_results(f))
    return rows


def aggregate(rows: list[dict], metric: str) -> dict[int, dict[str, list[float]]]:
    key = _MEAN_KEY[metric]
    agg: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        agg.setdefault(int(r["T"]), {"train": [], "val": []})[r["split"]].append(float(r[key]))
    return dict(sorted(agg.items()))


def plot_boxplots(agg, metric, out_path, adam: dict | None = None) -> None:
    ts = list(agg)
    fig, (ax_tr, ax_va) = plt.subplots(1, 2, sharey=True, figsize=(11, 4))
    for ax, split, title in ((ax_tr, "train", "train archs"), (ax_va, "val", "val archs")):
        ax.boxplot([agg[t][split] for t in ts], tick_labels=[str(t) for t in ts])
        ax.set_title(f"{title} — {_LABEL[metric]}")
        ax.set_xlabel("T (number of training architectures)")
        if adam is not None:
            vals = [adam[k][_ADAM_KEY[metric]] for k in adam]
            ax.axhline(float(np.median(vals)), ls="--", color="tab:red",
                       label="Adam median")
            ax.legend()
    ax_tr.set_ylabel(_LABEL[metric])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(agg, metric, out_path, adam: dict | None = None) -> None:
    ts = list(agg)
    xs = [float(np.mean(agg[t]["train"])) for t in ts]
    ys = [float(np.mean(agg[t]["val"])) for t in ts]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(xs, ys, "-o")
    for t, x, y in zip(ts, xs, ys):
        ax.annotate(f"T={t}", (x, y), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel(f"mean train-arch {_LABEL[metric]}")
    ax.set_ylabel(f"mean val-arch {_LABEL[metric]}")
    ax.set_title(f"Architecture scaling — {_LABEL[metric]}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
