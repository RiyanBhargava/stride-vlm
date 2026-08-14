#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=Path, default=Path("results/main/aggregate/results.csv")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/main/aggregate/accuracy_efficiency.pdf"),
    )
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install plotting dependencies: pip install -e '.[eval]'") from exc
    with args.csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No aggregate rows found in {args.csv}")
    models = sorted({row["model"] for row in rows})
    datasets = sorted({row["dataset"] for row in rows})
    methods = sorted({row["method"] for row in rows})
    colors = {
        method: plt.get_cmap("tab10")(index % 10) for index, method in enumerate(methods)
    }
    fig, axes = plt.subplots(
        len(models),
        len(datasets),
        figsize=(3.6 * len(datasets), 3.2 * len(models)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, model in enumerate(models):
        for column_index, dataset in enumerate(datasets):
            ax = axes[row_index][column_index]
            panel = [r for r in rows if r["model"] == model and r["dataset"] == dataset]
            for method in methods:
                selected = sorted(
                    (r for r in panel if r["method"] == method),
                    key=lambda r: float(r["latency_mean_s"]),
                )
                if not selected:
                    continue
                ax.plot(
                    [float(r["latency_mean_s"]) for r in selected],
                    [100 * float(r["score_mean"]) for r in selected],
                    marker="o",
                    color=colors[method],
                    label=method,
                )
            ax.set_title(dataset.upper())
            if column_index == 0:
                ax.set_ylabel(f"{model}\nTask score (%)")
            if row_index == len(models) - 1:
                ax.set_xlabel("Total latency (s/sample)")
            ax.grid(alpha=0.25)
    handles = [
        plt.Line2D([], [], color=colors[method], marker="o", label=method)
        for method in methods
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=min(len(methods), 6),
        frameon=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)


if __name__ == "__main__":
    main()
