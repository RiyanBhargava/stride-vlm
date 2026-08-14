#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def build_report(
    rows: list[dict[str, str]],
    paired: list[dict[str, str]],
    results_root: Path,
    reference: str,
) -> str:
    prediction_files = list(results_root.rglob("predictions.jsonl"))
    prediction_rows = sum(
        sum(1 for line in path.open(encoding="utf-8") if line.strip())
        for path in prediction_files
    )
    lines = [
        "# Current measured results",
        "",
        "> Generated from raw prediction JSONL and aggregate CSV files. Do not edit numerical "
        "values by hand.",
        "",
        "## Completeness",
        "",
        f"- Prediction files: {len(prediction_files):,}",
        f"- Prediction rows: {prediction_rows:,}",
        f"- Aggregate groups: {len(rows):,}",
        f"- Paired comparisons: {len(paired):,}",
        "",
        "## Macro-average by model",
        "",
        "| Model | Method | Budget | Mean score | Mean latency (ms) |",
        "|---|---|---:|---:|---:|",
    ]
    groups: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["method"], int(row["budget"]))].append(row)
    for (model, method, budget), group in sorted(groups.items()):
        lines.append(
            f"| {model} | {method} | {budget} | "
            f"{fmt(mean(float(item['score_mean']) for item in group))} | "
            f"{fmt(1000 * mean(float(item['latency_mean_s']) for item in group), 1)} |"
        )

    lines.extend(
        [
            "",
            f"## {reference} against dense and the strongest compressed baseline",
            "",
            "| Model | Dataset | Budget | Reference | Dense | Best compressed control | "
            "Ref. - best | Token reduction | Speedup vs dense |",
            "|---|---|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    reference_rows = sorted(
        (row for row in rows if row["method"] == reference),
        key=lambda row: (row["model"], row["dataset"], int(row["budget"])),
    )
    for row in reference_rows:
        dense = next(
            item
            for item in rows
            if item["model"] == row["model"]
            and item["dataset"] == row["dataset"]
            and item["method"] == "dense"
        )
        controls = [
            item
            for item in rows
            if item["model"] == row["model"]
            and item["dataset"] == row["dataset"]
            and item["budget"] == row["budget"]
            and item["method"] not in {"dense", reference}
        ]
        best = max(controls, key=lambda item: float(item["score_mean"]))
        reference_score = float(row["score_mean"])
        dense_score = float(dense["score_mean"])
        best_score = float(best["score_mean"])
        token_reduction = 1 - float(row["visual_tokens_mean"]) / float(
            dense["visual_tokens_mean"]
        )
        speedup = float(dense["latency_mean_s"]) / float(row["latency_mean_s"])
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['budget']} | "
            f"{fmt(reference_score)} | {fmt(dense_score)} | "
            f"{best['method']} ({fmt(best_score)}) | {fmt(reference_score - best_score)} | "
            f"{100 * token_reduction:.1f}% | {speedup:.2f}x |"
        )

    significant = [item for item in paired if float(item["p_holm"]) < 0.05]
    positive = [item for item in significant if float(item["score_difference"]) > 0]
    negative = [item for item in significant if float(item["score_difference"]) < 0]
    lines.extend(
        [
            "",
            "## Multiplicity-corrected paired findings",
            "",
            f"- Significant positive comparisons: {len(positive)}",
            f"- Significant negative comparisons: {len(negative)}",
            "",
            "| Direction | Model | Dataset | Budget | Comparison | Difference | 95% CI | "
            "Holm p |",
            "|---|---|---|---:|---|---:|---|---:|",
        ]
    )
    for item in sorted(
        significant,
        key=lambda value: (
            value["model"],
            value["dataset"],
            int(value["budget"]),
            value["comparison"],
        ),
    ):
        effect = float(item["score_difference"])
        lines.append(
            f"| {'positive' if effect > 0 else 'negative'} | {item['model']} | "
            f"{item['dataset']} | {item['budget']} | {item['comparison']} | "
            f"{fmt(effect)} | [{fmt(float(item['ci95_low']))}, "
            f"{fmt(float(item['ci95_high']))}] | {float(item['p_holm']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "A new router is not considered supported merely because it compresses tokens. It must "
            "beat the strongest matched-budget control with paired uncertainty, "
            "while retaining a measured latency advantage. Any method changed after inspecting "
            "final outcomes requires a new disjoint evaluation set.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=Path, default=Path("results/main/aggregate/results.csv")
    )
    parser.add_argument(
        "--paired-csv",
        type=Path,
        default=Path("results/main/aggregate/paired_comparisons.csv"),
    )
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/main/runs")
    )
    parser.add_argument("--reference", default="stride")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/main/aggregate/results_report.md"),
    )
    args = parser.parse_args()
    if args.reference == 'stride':
        args.reference = 'stride'
    report = build_report(
        read_csv(args.csv),
        read_csv(args.paired_csv),
        args.results_root,
        args.reference,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote measured result report to {args.output}")


if __name__ == "__main__":
    main()
