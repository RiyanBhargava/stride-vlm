#!/usr/bin/env python
"""Recompute stored scores and summaries directly from predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from stride.evaluation.metrics import EVALUATION_VERSION, score_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument(
        "--allow-unversioned",
        action="store_true",
        help="audit legacy rows but do not accept them as publication-ready",
    )
    args = parser.parse_args()
    prediction_files = sorted(args.root.rglob("predictions.jsonl"))
    errors: list[str] = []
    rows_checked = 0
    for path in prediction_files:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows_checked += len(records)
        for row in records:
            expected = score_prediction(
                row["prediction"], tuple(row["answers"]), row["metric"]
            )
            if abs(expected - float(row["score"])) > 1e-12:
                errors.append(
                    f"{path}: id={row['id']} stored={row['score']} "
                    f"recomputed={expected}"
                )
        summary_path = path.parent / "summary.json"
        if summary_path.exists() and records:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected_mean = mean(float(row["score"]) for row in records)
            if int(summary.get("samples", -1)) != len(records):
                errors.append(f"{summary_path}: sample count mismatch")
            if abs(float(summary.get("score")) - expected_mean) > 1e-12:
                errors.append(f"{summary_path}: score mean mismatch")
            if (
                summary.get("evaluation_version") != EVALUATION_VERSION
                and not args.allow_unversioned
            ):
                errors.append(
                    f"{summary_path}: evaluator version is missing or obsolete"
                )
    print(
        f"audited {rows_checked} rows in {len(prediction_files)} prediction files "
        f"with {EVALUATION_VERSION}"
    )
    if errors:
        print("\n".join(errors[:20]))
        if len(errors) > 20:
            print(f"... plus {len(errors) - 20} more errors")
        raise SystemExit("score audit failed")
    print("score audit passed: rows and summaries recompute exactly")


if __name__ == "__main__":
    main()
