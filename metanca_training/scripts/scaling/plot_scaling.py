# metanca_training/scripts/scaling/plot_scaling.py
"""CLI: render loss+accuracy boxplot & scatter figures for one ablation."""

import argparse
import json
from pathlib import Path

from metanca_training.scaling.plotting import (
    aggregate, load_results_dir, plot_boxplots, plot_scatter,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True,
                   help="per-ablation dir, e.g. results/scaling/fixed3")
    p.add_argument("--adam", default=None)
    p.add_argument("--outdir", default="results/figures")
    p.add_argument("--tag", default="fixed3")
    p.add_argument("--metrics", default="loss,acc",
                   help="comma-separated metrics to plot (default: loss,acc)")
    args = p.parse_args()

    rows = load_results_dir(args.results_dir)
    adam = json.loads(Path(args.adam).read_text()) if args.adam else None
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = args.metrics.split(",")
    for metric in metrics:
        agg = aggregate(rows, metric)
        plot_boxplots(agg, metric, outdir / f"{args.tag}_box_{metric}.png", adam)
        plot_scatter(agg, metric, outdir / f"{args.tag}_scatter_{metric}.png", adam)
    print(f"wrote figures to {outdir}")


if __name__ == "__main__":
    main()
