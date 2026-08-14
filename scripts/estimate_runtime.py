#!/usr/bin/env python
"""Estimate a target matrix runtime from completed real-model pilot summaries."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from stride.progress import format_duration


@dataclass(frozen=True)
class Timing:
    prefill_seconds: float
    generation_seconds: float
    max_new_tokens: float
    samples: int

    def projected_seconds(self, target_new_tokens: int) -> float:
        token_ratio = target_new_tokens / max(self.max_new_tokens, 1.0)
        return self.prefill_seconds + self.generation_seconds * token_ratio


def load_calibration(root: Path) -> dict[str, dict[str, Timing]]:
    totals: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"prefill": 0.0, "generation": 0.0, "tokens": 0.0, "samples": 0.0}
    )
    for path in root.rglob("summary.json"):
        relative = path.relative_to(root)
        if len(relative.parts) < 4:
            continue
        model = relative.parts[0]
        row = json.loads(path.read_text(encoding="utf-8"))
        if not row.get("samples") or row.get("mean_prefill_seconds") is None:
            continue
        kind = "dense" if row.get("method") == "dense" else "reduced"
        samples = int(row["samples"])
        target = totals[(model, kind)]
        target["prefill"] += float(row["mean_prefill_seconds"]) * samples
        target["generation"] += float(row["mean_generation_seconds"]) * samples
        target["tokens"] += float(row.get("generation_kwargs", {}).get("max_new_tokens", 1)) * samples
        target["samples"] += samples

    calibration: dict[str, dict[str, Timing]] = defaultdict(dict)
    for (model, kind), values in totals.items():
        samples = int(values["samples"])
        calibration[model][kind] = Timing(
            values["prefill"] / samples,
            values["generation"] / samples,
            values["tokens"] / samples,
            samples,
        )
    return dict(calibration)


def sample_count(spec: dict, benchmark: str) -> int:
    configured = spec.get("max_samples")
    data = Path(spec.get("data_root", "data")) / benchmark / "samples.jsonl"
    if data.exists():
        with data.open("r", encoding="utf-8") as handle:
            available = sum(1 for line in handle if line.strip())
        return min(available, configured) if configured else available
    if configured:
        return int(configured)
    raise FileNotFoundError(f"Cannot count samples for {benchmark}: {data} is missing")


def evaluation_counts(spec: dict) -> dict[str, int]:
    counts = {"dense": 0, "reduced": 0}
    for benchmark in spec["benchmarks"]:
        samples = sample_count(spec, benchmark)
        for method in spec["methods"]:
            if method == "dense":
                counts["dense"] += samples
                continue
            seeds = len(spec.get("random_seeds", [0])) if method == "random" else 1
            counts["reduced"] += samples * len(spec["budgets"]) * seeds
    return counts


def estimate_model_seconds(timings: dict[str, Timing], spec: dict) -> tuple[float, dict[str, int]]:
    missing = {"dense", "reduced"} - set(timings)
    if missing:
        raise ValueError(f"missing calibration for {sorted(missing)}")
    counts = evaluation_counts(spec)
    target_tokens = int(spec.get("max_new_tokens", 32))
    seconds = sum(
        counts[kind] * timings[kind].projected_seconds(target_tokens)
        for kind in ("dense", "reduced")
    )
    return seconds, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/main/preflight")
    )
    parser.add_argument(
        "--target-config", type=Path, default=Path("configs/experiments/few_hours.yaml")
    )
    parser.add_argument("--low-factor", type=float, default=0.8)
    parser.add_argument("--high-factor", type=float, default=1.6)
    args = parser.parse_args()
    if args.results_root == Path('results/main/preflight'):
        args.results_root = Path('tmp/smoke')
    if args.target_config == Path('configs/experiments/few_hours.yaml'):
        args.target_config = Path('configs/experiments/main.yaml')

    spec = yaml.safe_load(args.target_config.read_text(encoding="utf-8"))
    calibration = load_calibration(args.results_root)
    target_tokens = int(spec.get("max_new_tokens", 32))
    print(f"calibration: {args.results_root}; target: {args.target_config}")
    print(f"target generation limit: {target_tokens} token(s)")
    total = 0.0
    calibrated_models = 0
    for model in spec["models"]:
        timings = calibration.get(model, {})
        try:
            seconds, counts = estimate_model_seconds(timings, spec)
        except ValueError as exc:
            print(f"{model}: unavailable ({exc}); complete its preflight first")
            continue
        dense = timings["dense"].projected_seconds(target_tokens)
        reduced = timings["reduced"].projected_seconds(target_tokens)
        print(
            f"{model}: {counts['dense'] + counts['reduced']:,} evaluations, "
            f"projected dense={dense:.3f}s/sample, reduced={reduced:.3f}s/sample, "
            f"central ETA={format_duration(seconds)}"
        )
        total += seconds
        calibrated_models += 1

    if not calibrated_models:
        raise SystemExit("No complete model calibration was found")
    label = "full-matrix" if calibrated_models == len(spec["models"]) else "calibrated-model partial"
    print(
        f"{label} ETA: {format_duration(total * args.low_factor)} to "
        f"{format_duration(total * args.high_factor)} "
        "(cached checkpoints; conservative range, not a guarantee)"
    )
    if calibrated_models != len(spec["models"]):
        print("Full ETA will be available after every target model completes the preflight.")


if __name__ == "__main__":
    main()
