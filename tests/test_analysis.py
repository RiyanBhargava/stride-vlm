import json

import pytest

from stride.analysis import (
    aggregate_ablation_study,
    aggregate_runs,
    bootstrap_ci,
    paired_comparisons,
    write_latex_table,
)


def test_bootstrap_reproducible():
    assert bootstrap_ci([0.0, 1.0, 1.0], seed=4, samples=200) == bootstrap_ci(
        [0.0, 1.0, 1.0], seed=4, samples=200
    )


def test_aggregate_and_latex(tmp_path):
    run = tmp_path / "model" / "pope" / "stride_b8"
    run.mkdir(parents=True)
    rows = [
        {"id": "a", "method": "stride", "budget": 8, "seed": 0, "score": 1.0, "generation_seconds": 0.2, "output_visual_tokens": 8},
        {"id": "b", "method": "stride", "budget": 8, "seed": 0, "score": 0.0, "generation_seconds": 0.4, "output_visual_tokens": 8},
    ]
    (run / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    aggregate = aggregate_runs(tmp_path, tmp_path / "results.csv")
    assert len(aggregate) == 1
    assert aggregate[0]["model"] == "model"
    assert aggregate[0]["dataset"] == "pope"
    assert aggregate[0]["score_mean"] == 0.5
    assert aggregate[0]["seeds"] == 1
    assert aggregate[0]["prefill_mean_s"] == 0.0
    assert aggregate[0]["generation_mean_s"] == pytest.approx(0.3)
    assert aggregate[0]["latency_mean_s"] == pytest.approx(0.3)
    write_latex_table(aggregate, tmp_path / "results.tex")
    assert "stride" in (tmp_path / "results.tex").read_text(encoding="utf-8")


def test_latex_bolds_only_best_budget_matched_compressed_method(tmp_path):
    common = {
        "model": "model_a",
        "dataset": "pope",
        "score_ci95_low": 0.4,
        "score_ci95_high": 0.6,
        "latency_mean_s": 0.2,
        "samples": 10,
    }
    rows = [
        {**common, "method": "dense", "budget": 576, "score_mean": 0.9},
        {**common, "method": "pool", "budget": 64, "score_mean": 0.5},
        {**common, "method": "salience_coverage", "budget": 64, "score_mean": 0.6},
    ]
    output = tmp_path / "results.tex"
    write_latex_table(rows, output)
    latex = output.read_text(encoding="utf-8")
    assert r"model\_a" in latex
    assert r"salience\_coverage" in latex
    assert r"\textbf{60.00 $\pm$ 10.00}" in latex
    assert r"\textbf{90.00" not in latex


def test_aggregate_keeps_models_separate(tmp_path):
    for model, tokens in (("model-a", 576), ("model-b", 1024)):
        run = tmp_path / model / "pope" / "dense_b576"
        run.mkdir(parents=True)
        row = {
            "id": "same-id",
            "method": "dense",
            "budget": 576,
            "seed": 0,
            "score": 1.0,
            "prefill_seconds": 0.1,
            "generation_seconds": 0.2,
            "output_visual_tokens": tokens,
        }
        (run / "predictions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    aggregate = aggregate_runs(tmp_path, tmp_path / "results.csv")
    assert len(aggregate) == 2
    assert {row["model"] for row in aggregate} == {"model-a", "model-b"}
    assert {row["visual_tokens_mean"] for row in aggregate} == {576, 1024}


def test_paired_comparisons_use_shared_sample_ids(tmp_path):
    for method, scores in (("stride", {"a": 1.0, "b": 1.0}), ("pool", {"a": 0.0, "b": 1.0})):
        run = tmp_path / "model" / "pope" / f"{method}_b8"
        run.mkdir(parents=True)
        rows = [
            {
                "id": sample_id,
                "method": method,
                "budget": 8,
                "seed": 0,
                "score": score,
            }
            for sample_id, score in scores.items()
        ]
        (run / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    comparisons = paired_comparisons(tmp_path, tmp_path / "paired.csv")
    assert len(comparisons) == 1
    assert comparisons[0]["comparison"] == "pool"
    assert comparisons[0]["score_difference"] == 0.5
    assert comparisons[0]["samples"] == 2


def test_ablation_study_records_paired_effects(tmp_path):
    for variant, scores in (("full", [1.0, 1.0]), ("no_relay", [0.0, 1.0])):
        run = tmp_path / variant
        run.mkdir()
        rows = [
            {
                "id": sample_id,
                "score": score,
                "generation_seconds": 0.2,
                "routing_seconds": 0.01,
                "output_visual_tokens": 8,
            }
            for sample_id, score in zip(("a", "b"), scores)
        ]
        (run / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    aggregate = aggregate_ablation_study(tmp_path)
    by_name = {row["variant"]: row for row in aggregate}
    assert by_name["full"]["difference_vs_full"] == 0.0
    assert by_name["no_relay"]["difference_vs_full"] == -0.5
    assert (tmp_path / "ablation_results.csv").exists()


def test_ablation_study_can_reuse_external_full_predictions(tmp_path):
    study = tmp_path / "study"
    variant = study / "no_relay"
    variant.mkdir(parents=True)
    reference = tmp_path / "main_stride_predictions.jsonl"
    full_rows = [
        {
            "id": sample_id,
            "score": score,
            "total_seconds": 0.2,
            "routing_seconds": 0.01,
            "output_visual_tokens": 8,
        }
        for sample_id, score in (("a", 1.0), ("b", 1.0))
    ]
    variant_rows = [
        {**row, "score": score}
        for row, score in zip(full_rows, (0.0, 1.0))
    ]
    reference.write_text(
        "".join(json.dumps(row) + "\n" for row in full_rows), encoding="utf-8"
    )
    (variant / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in variant_rows),
        encoding="utf-8",
    )
    rows = aggregate_ablation_study(study, reference_predictions=reference)
    by_name = {row["variant"]: row for row in rows}
    assert by_name["full"]["score_mean"] == 1.0
    assert by_name["no_relay"]["difference_vs_full"] == -0.5
