#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import time
from pathlib import Path
from statistics import mean

import torch
import yaml
from tqdm import tqdm

from stride.analysis import aggregate_ablation_study
from stride.config import RouterConfig
from stride.evaluation.data import read_jsonl
from stride.evaluation.runner import evaluate_jsonl, format_evaluation_prompt
from stride.progress import configure_library_output, format_duration, timed_stage

try:
    from scripts.run_ablations import STRIDE_ABLATIONS
    from scripts.run_matrix import load_adapter
except ImportError:
    from run_ablations import STRIDE_ABLATIONS
    from run_matrix import load_adapter


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_records(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_full_reference(
    data_path: Path,
    predictions_path: Path,
    summary_path: Path,
    budget: int,
    maximum: int | None,
    expected_config: RouterConfig,
    max_new_tokens: int,
) -> dict:
    if not predictions_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f'missing completed main STRIDE reference: {predictions_path}'
        )
    examples = list(read_jsonl(data_path))
    if maximum is not None:
        examples = examples[:maximum]
    expected_ids = [example.sample_id for example in examples]
    records = read_records(predictions_path)
    found_ids = [str(record['id']) for record in records]
    if len(found_ids) != len(set(found_ids)):
        raise ValueError(f'duplicate IDs in {predictions_path}')
    if found_ids != expected_ids:
        raise ValueError(
            f'main STRIDE reference does not exactly match {data_path}: '
            f'expected={len(expected_ids)}, found={len(found_ids)}'
        )
    if any(
        record.get('method') != 'stride'
        or int(record.get('budget', -1)) != budget
        for record in records
    ):
        raise ValueError(f'reference is not uniformly stride@{budget}')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('method') != 'stride' or int(summary.get('budget', -1)) != budget:
        raise ValueError(f'mismatched main summary: {summary_path}')
    expected = RouterConfig.from_dict(
        {**expected_config.to_dict(), 'budget': budget}
    ).to_dict()
    if summary.get('router_config') != expected:
        raise ValueError(f'main STRIDE config differs: {summary_path}')
    if int(summary.get('samples', -1)) != len(expected_ids):
        raise ValueError(f'incomplete main summary: {summary_path}')
    actual_tokens = summary.get('generation_kwargs', {}).get('max_new_tokens', -1)
    if int(actual_tokens) != max_new_tokens:
        raise ValueError(f'generation settings differ: {summary_path}')
    return {
        'predictions': str(predictions_path),
        'summary': str(summary_path),
        'sha256': file_sha256(predictions_path),
        'samples': len(records),
        'score': mean(float(record['score']) for record in records),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for variant in sorted({str(row['variant']) for row in rows}):
        selected = [row for row in rows if row['variant'] == variant]
        differences = [float(row['difference_vs_full']) for row in selected]
        output.append(
            {
                'variant': variant,
                'cells': len(selected),
                'mean_score': mean(float(row['score_mean']) for row in selected),
                'mean_difference_vs_full': mean(differences),
                'cells_higher': sum(value > 0 for value in differences),
                'cells_tied': sum(value == 0 for value in differences),
                'cells_lower': sum(value < 0 for value in differences),
                'significant_lower': sum(
                    float(row['p_holm']) < 0.05
                    and float(row['difference_vs_full']) < 0
                    for row in selected
                ),
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/experiments/ablations.yaml'),
    )
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--dtype', default='bfloat16', choices=['float16', 'bfloat16', 'float32']
    )
    parser.add_argument('--attn-implementation', default=None)
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--show-library-warnings', action='store_true')
    parser.add_argument('--verbose-progress', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_library_output(args.show_library_warnings)
    spec = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    base = RouterConfig.from_yaml(spec['router_config'])
    variants = list(spec['variants'])
    unknown = set(variants) - set(STRIDE_ABLATIONS)
    if unknown:
        raise ValueError(f'unknown ablations: {sorted(unknown)}')
    models = list(spec['models'])
    benchmarks = list(spec['benchmarks'])
    default_budgets = [int(value) for value in spec['budgets']]
    budgets_by_model = {
        model: [
            int(value) for value in spec.get('budgets_by_model', {}).get(
                model, default_budgets
            )
        ]
        for model in models
    }
    maximum = spec.get('max_samples')
    max_new_tokens = int(spec.get('max_new_tokens', 12))
    data_root = Path(spec['data_root'])
    main_root = Path(spec['main_results_root'])
    output_root = Path(spec['output_root'])
    jobs = [
        (model, benchmark, budget, variant)
        for model in models
        for benchmark in benchmarks
        for budget in budgets_by_model[model]
        for variant in variants
    ]
    evaluations = len(jobs) * int(maximum or 0)
    tqdm.write(
        f'full ablations: {len(models)} model(s), {len(jobs)} run(s), '
        f'{evaluations:,} sample evaluations'
    )
    if evaluations:
        tqdm.write(
            f'rough serial ETA {format_duration(evaluations * 0.2)} to '
            f'{format_duration(evaluations * 0.8)}; live ETA replaces this'
        )
    if args.dry_run:
        for model, benchmark, budget, variant in jobs:
            print(f'{model}/{benchmark}/stride@{budget}/{variant}')
        return

    references: dict[tuple[str, str, int], dict] = {}
    for model in models:
        for benchmark in benchmarks:
            data_path = data_root / benchmark / 'samples.jsonl'
            for budget in budgets_by_model[model]:
                run_root = main_root / model / benchmark / f'stride_b{budget}'
                references[(model, benchmark, budget)] = validate_full_reference(
                    data_path,
                    run_root / 'predictions.jsonl',
                    run_root / 'summary.json',
                    budget,
                    maximum,
                    base,
                    max_new_tokens,
                )
    tqdm.write(f'validated {len(references)} exact main STRIDE references')
    if args.validate_only:
        return

    quantized = args.load_in_4bit or bool(spec.get('load_in_4bit', False))
    completed = 0
    started = time.perf_counter()
    combined: list[dict] = []
    with tqdm(
        total=len(jobs),
        desc='full ablations',
        unit='run',
        dynamic_ncols=True,
        mininterval=1.0,
    ) as bar:
        for model_index, model in enumerate(models, 1):
            tqdm.write(f'model {model_index}/{len(models)}: {model}')
            adapter = load_adapter(
                model,
                base,
                args.device,
                args.dtype,
                args.attn_implementation,
                quantized,
            )
            warmups = int(spec.get('warmup_samples', 1))
            if warmups:
                examples = list(
                    read_jsonl(data_root / benchmarks[0] / 'samples.jsonl')
                )[:warmups]
                with timed_stage(f'warm up {model} ({len(examples)} sample(s))'):
                    for example in examples:
                        adapter.generate(
                            example.image,
                            format_evaluation_prompt(example.question, example.metric),
                            method='dense',
                            budget=budgets_by_model[model][0],
                            routing_prompt=example.question,
                            max_new_tokens=min(4, max_new_tokens),
                        )
            for benchmark in benchmarks:
                data_path = data_root / benchmark / 'samples.jsonl'
                for budget in budgets_by_model[model]:
                    study_root = output_root / model / benchmark / f'stride_b{budget}'
                    study_root.mkdir(parents=True, exist_ok=True)
                    reference = references[(model, benchmark, budget)]
                    manifest_path = study_root / 'ablation_manifest.json'
                    manifest = {
                        'model': model,
                        'dataset': benchmark,
                        'method': 'stride',
                        'budget': budget,
                        'max_samples': maximum,
                        'max_new_tokens': max_new_tokens,
                        'base_router_config': base.to_dict(),
                        'full_reference': reference,
                        'full_is_reused_from_main': True,
                        'variants': {
                            name: STRIDE_ABLATIONS[name] for name in variants
                        },
                        'completed': [],
                    }
                    if manifest_path.exists():
                        prior = json.loads(
                            manifest_path.read_text(encoding='utf-8')
                        )
                        manifest['completed'] = prior.get('completed', [])
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2), encoding='utf-8'
                    )
                    for variant in variants:
                        config = RouterConfig.from_dict(
                            {
                                **base.to_dict(),
                                **STRIDE_ABLATIONS[variant],
                                'budget': budget,
                            }
                        )
                        adapter.router_config = config
                        label = f'{model}/{benchmark}/{budget}/{variant}'
                        bar.set_postfix_str(label)
                        if args.verbose_progress:
                            tqdm.write(
                                f'[ablation {completed + 1}/{len(jobs)}] {label}'
                            )
                        evaluate_jsonl(
                            adapter,
                            data_path,
                            study_root / variant,
                            'stride',
                            budget,
                            maximum,
                            True,
                            {'max_new_tokens': max_new_tokens},
                            config.seed,
                            progress_position=1,
                        )
                        if variant not in manifest['completed']:
                            manifest['completed'].append(variant)
                        manifest_path.write_text(
                            json.dumps(manifest, indent=2), encoding='utf-8'
                        )
                        completed += 1
                        bar.update(1)
                    cell_rows = aggregate_ablation_study(
                        study_root,
                        reference_predictions=Path(reference['predictions']),
                    )
                    combined.extend(
                        {
                            'model': model,
                            'dataset': benchmark,
                            'budget': budget,
                            **row,
                        }
                        for row in cell_rows
                    )
            del adapter
            gc.collect()
            if torch.cuda.is_available():
                with timed_stage('release CUDA cache'):
                    torch.cuda.empty_cache()

    write_csv(combined, output_root / 'ablation_results.csv')
    summary = summarize(combined)
    write_csv(summary, output_root / 'ablation_summary.csv')
    tqdm.write(
        f'full ablations complete in '
        f'{format_duration(time.perf_counter() - started)}; '
        f'wrote {len(combined)} cell rows'
    )


if __name__ == '__main__':
    main()
