#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from stride.config import RouterConfig
from stride.evaluation.data import read_jsonl
from stride.evaluation.runner import evaluate_jsonl, format_evaluation_prompt
from stride.models.llava import LlavaAdapter, MODEL_REGISTRY as LLAVA_MODELS
from stride.progress import configure_library_output, format_duration, timed_stage
from stride.routing import METHODS


@dataclass(frozen=True)
class Job:
    benchmark: str
    method: str
    budget: int
    seed: int
    output: Path
    data: Path


def build_jobs(spec: dict, model: str, base_seed: int) -> list[Job]:
    jobs: list[Job] = []
    dense_budget = spec.get('dense_budgets', {}).get(
        model, spec.get('dense_budget', 576)
    )
    unknown = set(spec['methods']) - set(METHODS)
    if unknown:
        raise ValueError(f'unknown methods in experiment: {sorted(unknown)}')
    for benchmark in spec['benchmarks']:
        data = Path(spec.get('data_root', 'data_holdout')) / benchmark / 'samples.jsonl'
        for method in spec['methods']:
            compressed = spec.get('budgets_by_model', {}).get(
                model, spec['budgets']
            )
            budgets = [dense_budget] if method == 'dense' else compressed
            seeds = (
                spec.get('random_seeds', [base_seed])
                if method == 'random'
                else [base_seed]
            )
            for budget in budgets:
                for seed in seeds:
                    suffix = f'_s{seed}' if len(seeds) > 1 else ''
                    output = (
                        Path(spec.get('output_root', 'results/main'))
                        / model
                        / benchmark
                        / f'{method}_b{budget}{suffix}'
                    )
                    jobs.append(Job(benchmark, method, budget, seed, output, data))
    return jobs


def load_adapter(
    alias: str,
    config: RouterConfig,
    device: str,
    dtype: str,
    attention: str | None,
    quantized: bool,
):
    return LlavaAdapter(
        LLAVA_MODELS.get(alias, alias),
        config,
        device=device,
        dtype=dtype,
        attn_implementation=attention,
        load_in_4bit=quantized,
    )


def count_evaluations(
    jobs: dict[str, list[Job]], maximum: int | None, sample_offset: int = 0
) -> int:
    counts: dict[Path, int] = {}
    total = 0
    for model_jobs in jobs.values():
        for job in model_jobs:
            if job.data.exists() and job.data not in counts:
                with job.data.open('r', encoding='utf-8') as handle:
                    counts[job.data] = sum(bool(line.strip()) for line in handle)
            if job.data in counts:
                available = max(0, counts[job.data] - sample_offset)
                total += min(available, maximum) if maximum else available
            elif maximum:
                total += maximum
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=Path, default=Path('configs/experiments/main.yaml')
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--dtype', default='bfloat16', choices=['float16', 'bfloat16', 'float32']
    )
    parser.add_argument('--attn-implementation', default=None)
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--seconds-per-sample-low', type=float, default=0.2)
    parser.add_argument('--seconds-per-sample-high', type=float, default=0.8)
    parser.add_argument('--show-library-warnings', action='store_true')
    parser.add_argument('--verbose-progress', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_library_output(args.show_library_warnings)
    spec = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    maximum = args.max_samples if args.max_samples is not None else spec.get('max_samples')
    sample_offset = int(spec.get('sample_offset', 0))
    base = RouterConfig.from_yaml(
        spec.get('router_config', 'configs/router/stride.yaml')
    )
    jobs = {model: build_jobs(spec, model, base.seed) for model in spec['models']}
    run_count = sum(len(value) for value in jobs.values())
    evaluations = count_evaluations(jobs, maximum, sample_offset)
    maximum_label = maximum if maximum is not None else 'full dataset'
    tqdm.write(
        f'matrix: {len(jobs)} model(s), {run_count} run(s), '
        f'max_samples={maximum_label}'
    )
    if evaluations:
        low = evaluations * args.seconds_per_sample_low
        high = evaluations * args.seconds_per_sample_high
        tqdm.write(
            f'planned sample evaluations: {evaluations:,}; rough serial ETA '
            f'{format_duration(low)} to {format_duration(high)} '
            '(downloads and model loading excluded; live ETA replaces this)'
        )
    if args.dry_run:
        for model, model_jobs in jobs.items():
            for job in model_jobs:
                print(
                    f'{model}: {job.benchmark} {job.method}@{job.budget} '
                    f'seed={job.seed} -> {job.output}'
                )
        return

    quantized = args.load_in_4bit or bool(spec.get('load_in_4bit', False))
    started = time.perf_counter()
    completed = 0
    with tqdm(
        total=run_count,
        desc='matrix jobs',
        unit='run',
        dynamic_ncols=True,
        mininterval=1.0,
    ) as bar:
        for model_index, (model, model_jobs) in enumerate(jobs.items(), 1):
            tqdm.write(f'model {model_index}/{len(jobs)}: {model}')
            try:
                adapter = load_adapter(
                    model,
                    base,
                    args.device,
                    args.dtype,
                    args.attn_implementation,
                    quantized,
                )
            except OSError as error:
                message = str(error).lower()
                if 'gated repo' in message or 'access to model' in message:
                    raise SystemExit(
                        f'access denied for gated checkpoint {model}; accept its '
                        'Hugging Face license with this Python account'
                    ) from None
                raise
            warmups = int(spec.get('warmup_samples', 1))
            if warmups and model_jobs:
                examples = list(read_jsonl(model_jobs[0].data))[:warmups]
                with timed_stage(f'warm up {model} ({len(examples)} sample(s))'):
                    for example in examples:
                        adapter.generate(
                            example.image,
                            format_evaluation_prompt(example.question, example.metric),
                            method='dense',
                            budget=spec.get('dense_budgets', {}).get(
                                model, spec.get('dense_budget', 576)
                            ),
                            routing_prompt=example.question,
                            max_new_tokens=min(4, spec.get('max_new_tokens', 12)),
                        )
            for job in model_jobs:
                if not job.data.exists():
                    raise FileNotFoundError(
                        f'missing {job.data}; prepare the benchmark first'
                    )
                adapter.router_config = RouterConfig.from_dict(
                    {**base.to_dict(), 'budget': job.budget, 'seed': job.seed}
                )
                label = f'{model}/{job.benchmark}/{job.method}@{job.budget}/s{job.seed}'
                bar.set_postfix_str(label)
                if args.verbose_progress:
                    tqdm.write(f'[run {completed + 1}/{run_count}] {label}')
                run_started = time.perf_counter()
                evaluate_jsonl(
                    adapter,
                    job.data,
                    job.output,
                    job.method,
                    job.budget,
                    maximum,
                    True,
                    {'max_new_tokens': spec.get('max_new_tokens', 12)},
                    job.seed,
                    progress_position=1,
                    sample_offset=sample_offset,
                )
                completed += 1
                bar.update(1)
                bar.set_postfix_str(
                    f'{label}, last={format_duration(time.perf_counter() - run_started)}'
                )
            del adapter
            gc.collect()
            if torch.cuda.is_available():
                with timed_stage('release CUDA cache'):
                    torch.cuda.empty_cache()
    tqdm.write(f'matrix complete in {format_duration(time.perf_counter() - started)}')
    target = spec.get('estimate_target_config')
    if target:
        tqdm.write('hardware-calibrated estimate for the main experiment:')
        subprocess.run(
            [
                sys.executable,
                'scripts/estimate_runtime.py',
                '--results-root',
                str(spec.get('output_root', 'results/smoke')),
                '--target-config',
                str(target),
            ],
            check=True,
        )


if __name__ == '__main__':
    main()
