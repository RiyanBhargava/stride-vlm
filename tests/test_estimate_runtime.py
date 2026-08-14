import json

from scripts.estimate_runtime import evaluation_counts, estimate_model_seconds, load_calibration


def test_runtime_estimate_from_summaries(tmp_path):
    run = tmp_path / "model-a" / "pope" / "stride_b64"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "method": "stride",
                "samples": 2,
                "mean_prefill_seconds": 0.2,
                "mean_generation_seconds": 0.3,
                "generation_kwargs": {"max_new_tokens": 8},
            }
        ),
        encoding="utf-8",
    )
    dense = tmp_path / "model-a" / "pope" / "dense_b576"
    dense.mkdir(parents=True)
    (dense / "summary.json").write_text(
        json.dumps(
            {
                "method": "dense",
                "samples": 2,
                "mean_prefill_seconds": 0.1,
                "mean_generation_seconds": 0.4,
                "generation_kwargs": {"max_new_tokens": 8},
            }
        ),
        encoding="utf-8",
    )
    spec = {
        "benchmarks": ["pope"],
        "methods": ["dense", "random", "stride"],
        "random_seeds": [0, 1, 2],
        "budgets": [64],
        "max_samples": 10,
        "max_new_tokens": 16,
        "data_root": tmp_path / "missing-data",
    }
    timings = load_calibration(tmp_path)["model-a"]
    assert evaluation_counts(spec) == {"dense": 10, "reduced": 40}
    seconds, _ = estimate_model_seconds(timings, spec)
    assert seconds == 10 * 0.9 + 40 * 0.8
