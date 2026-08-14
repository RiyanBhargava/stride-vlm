from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def bootstrap_ci(values: list[float], seed: int = 0, samples: int = 10_000) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return estimates[int(0.025 * samples)], estimates[int(0.975 * samples)]


def paired_bootstrap(
    differences: list[float], seed: int = 0, samples: int = 10_000
) -> tuple[float, float, float, float]:
    """Mean effect, paired bootstrap interval, and sign-flip permutation p-value."""
    if not differences:
        return math.nan, math.nan, math.nan, math.nan
    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choices(differences, k=len(differences))) for _ in range(samples)
    )
    low, high = estimates[int(0.025 * samples)], estimates[int(0.975 * samples)]
    observed = mean(differences)
    permutation_rng = random.Random(seed + 1)
    extreme = 0
    for _ in range(samples):
        permuted = mean(
            value if permutation_rng.random() < 0.5 else -value for value in differences
        )
        extreme += abs(permuted) >= abs(observed) - 1e-12
    p_value = (extreme + 1) / (samples + 1)
    return observed, low, high, p_value


def aggregate_runs(root: str | Path, output_csv: str | Path) -> list[dict[str, Any]]:
    source = Path(root)
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for prediction_path in source.rglob("predictions.jsonl"):
        relative = prediction_path.relative_to(source)
        if len(relative.parts) < 4:
            raise ValueError(
                f"Expected results/<model>/<dataset>/<run>/predictions.jsonl, got {relative}"
            )
        model, dataset = relative.parts[0], relative.parts[1]
        with prediction_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                grouped[(model, dataset, row["method"], int(row["budget"]))].append(row)
    output_rows: list[dict[str, Any]] = []
    for (model, dataset, method, budget), rows in sorted(grouped.items()):
        # Average stochastic repetitions by sample before bootstrapping questions.
        # Otherwise five random seeds would incorrectly quintuple the sample size.
        sample_scores: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            sample_scores[str(row["id"])].append(float(row["score"]))
        scores = [mean(values) for values in sample_scores.values()]
        low, high = bootstrap_ci(scores)
        prefill = [float(r.get("prefill_seconds", 0.0)) for r in rows]
        generation = [float(r["generation_seconds"]) for r in rows]
        preprocessing = [float(r.get("preprocessing_seconds", 0.0)) for r in rows]
        vision = [float(r.get("vision_seconds", 0.0)) for r in rows]
        routing = [float(r.get("routing_seconds", 0.0)) for r in rows]
        packing = [float(r.get("packing_seconds", 0.0)) for r in rows]
        total_latency = [
            float(r.get("total_seconds", r.get("prefill_seconds", 0.0) + r["generation_seconds"]))
            for r in rows
        ]
        output_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "method": method,
                "budget": budget,
                "samples": len(sample_scores),
                "seeds": len({int(r.get("seed", 0)) for r in rows}),
                "score_mean": mean(scores),
                "score_std": stdev(scores) if len(scores) > 1 else 0.0,
                "score_ci95_low": low,
                "score_ci95_high": high,
                "prefill_mean_s": mean(prefill),
                "preprocessing_mean_s": mean(preprocessing),
                "vision_mean_s": mean(vision),
                "routing_mean_s": mean(routing),
                "packing_mean_s": mean(packing),
                "generation_mean_s": mean(generation),
                "latency_mean_s": mean(total_latency),
                "visual_tokens_mean": mean(float(r["output_visual_tokens"]) for r in rows),
            }
        )
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if output_rows:
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    return output_rows


def paired_comparisons(
    root: str | Path,
    output_csv: str | Path,
    reference: str = "stride",
) -> list[dict[str, Any]]:
    """Write sample-ID-paired effects for the reference router against every method."""
    source = Path(root)
    groups: dict[tuple[str, str, str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for prediction_path in source.rglob("predictions.jsonl"):
        relative = prediction_path.relative_to(source)
        if len(relative.parts) < 4:
            continue
        model, dataset = relative.parts[0], relative.parts[1]
        with prediction_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (model, dataset, str(row["method"]), int(row["budget"]))
                groups[key][str(row["id"])].append(float(row["score"]))

    collapsed = {
        key: {sample_id: mean(values) for sample_id, values in sample_scores.items()}
        for key, sample_scores in groups.items()
    }
    output_rows: list[dict[str, Any]] = []
    reference_groups = [key for key in collapsed if key[2] == reference]
    for model, dataset, _, budget in sorted(reference_groups):
        reference_scores = collapsed[(model, dataset, reference, budget)]
        candidates = [
            key
            for key in collapsed
            if key[0] == model
            and key[1] == dataset
            and key[2] != reference
            and (key[2] == "dense" or key[3] == budget)
        ]
        family_rows: list[dict[str, Any]] = []
        for candidate in sorted(candidates, key=lambda key: (key[2], key[3])):
            comparison_scores = collapsed[candidate]
            shared = sorted(set(reference_scores) & set(comparison_scores))
            differences = [reference_scores[item] - comparison_scores[item] for item in shared]
            effect, low, high, p_value = paired_bootstrap(differences)
            family_rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "budget": budget,
                    "reference": reference,
                    "comparison": candidate[2],
                    "comparison_budget": candidate[3],
                    "samples": len(shared),
                    "score_difference": effect,
                    "ci95_low": low,
                    "ci95_high": high,
                    "win_rate": mean(float(value > 0) for value in differences),
                    "tie_rate": mean(float(value == 0) for value in differences),
                    "p_value": p_value,
                    "p_holm": p_value,
                }
            )
        ordered = sorted(range(len(family_rows)), key=lambda i: family_rows[i]["p_value"])
        adjusted = 0.0
        for rank, index in enumerate(ordered):
            raw = float(family_rows[index]["p_value"])
            adjusted = max(adjusted, min(1.0, raw * (len(ordered) - rank)))
            family_rows[index]["p_holm"] = adjusted
        output_rows.extend(family_rows)

    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if output_rows:
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    return output_rows


def aggregate_ablation_study(
    root: str | Path,
    output_csv: str | Path | None = None,
    reference: str = "full",
    reference_predictions: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Aggregate a structured ``run_ablations.py`` study with paired effects."""
    source = Path(root)
    variants: dict[str, list[dict[str, Any]]] = {}
    for prediction_path in source.glob("*/predictions.jsonl"):
        with prediction_path.open("r", encoding="utf-8") as handle:
            variants[prediction_path.parent.name] = [
                json.loads(line) for line in handle if line.strip()
            ]
    if reference_predictions is not None:
        with Path(reference_predictions).open("r", encoding="utf-8") as handle:
            variants[reference] = [
                json.loads(line) for line in handle if line.strip()
            ]
    if reference not in variants:
        raise ValueError(f"Missing reference ablation '{reference}' under {source}")

    collapsed: dict[str, dict[str, float]] = {}
    for name, records in variants.items():
        scores: dict[str, list[float]] = defaultdict(list)
        for record in records:
            scores[str(record["id"])].append(float(record["score"]))
        collapsed[name] = {sample_id: mean(values) for sample_id, values in scores.items()}

    reference_scores = collapsed[reference]
    output_rows: list[dict[str, Any]] = []
    for name in sorted(variants, key=lambda value: (value != reference, value)):
        records = variants[name]
        scores = list(collapsed[name].values())
        score_low, score_high = bootstrap_ci(scores)
        shared = sorted(set(reference_scores) & set(collapsed[name]))
        differences = [collapsed[name][item] - reference_scores[item] for item in shared]
        if name == reference:
            effect, difference_low, difference_high, p_value = 0.0, 0.0, 0.0, 1.0
        else:
            effect, difference_low, difference_high, p_value = paired_bootstrap(differences)
        latencies = [
            float(
                record.get(
                    "total_seconds",
                    record.get("prefill_seconds", 0.0)
                    + record.get("generation_seconds", 0.0),
                )
            )
            for record in records
        ]
        output_rows.append(
            {
                "variant": name,
                "samples": len(scores),
                "score_mean": mean(scores),
                "score_ci95_low": score_low,
                "score_ci95_high": score_high,
                "difference_vs_full": effect,
                "difference_ci95_low": difference_low,
                "difference_ci95_high": difference_high,
                "p_value": p_value,
                "p_holm": p_value,
                "latency_mean_s": mean(latencies),
                "routing_mean_s": mean(
                    float(record.get("routing_seconds", 0.0)) for record in records
                ),
                "visual_tokens_mean": mean(
                    float(record["output_visual_tokens"]) for record in records
                ),
            }
        )

    tested = [index for index, row in enumerate(output_rows) if row["variant"] != reference]
    tested.sort(key=lambda index: float(output_rows[index]["p_value"]))
    adjusted = 0.0
    for rank, index in enumerate(tested):
        raw = float(output_rows[index]["p_value"])
        adjusted = max(adjusted, min(1.0, raw * (len(tested) - rank)))
        output_rows[index]["p_holm"] = adjusted

    destination = Path(output_csv) if output_csv else source / "ablation_results.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if output_rows:
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    return output_rows


def write_latex_table(rows: list[dict[str, Any]], path: str | Path) -> None:
    def escape(value: Any) -> str:
        return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_")

    best_compressed: dict[tuple[str, str, int], float] = {}
    for row in rows:
        if row["method"] == "dense":
            continue
        key = (row["model"], row["dataset"], int(row["budget"]))
        best_compressed[key] = max(
            best_compressed.get(key, -math.inf), float(row["score_mean"])
        )

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Method & Budget & Score $\uparrow$ & Latency (s) $\downarrow$ & $N$ \\",
        r"\midrule",
    ]
    lines[0] = lines[0].replace("{llrrrr}", "{lllrrrr}")
    lines[2] = lines[2].replace("Dataset", "Model & Dataset", 1).replace(
        "Latency (s)", "Total latency (s)"
    )
    for row in rows:
        score = 100 * float(row["score_mean"])
        ci = 50 * (float(row["score_ci95_high"]) - float(row["score_ci95_low"]))
        score_text = f"{score:.2f} $\\pm$ {ci:.2f}"
        key = (row["model"], row["dataset"], int(row["budget"]))
        if (
            row["method"] != "dense"
            and math.isclose(float(row["score_mean"]), best_compressed[key])
        ):
            score_text = rf"\textbf{{{score_text}}}"
        lines.append(
            f'{escape(row["model"])} & '
            f'{escape(row["dataset"])} & {escape(row["method"])} & {row["budget"]} & '
            f'{score_text} & {float(row["latency_mean_s"]):.3f} & {row["samples"]} \\\\'
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
