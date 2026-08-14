from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .analysis import aggregate_runs, paired_comparisons, write_latex_table
from .config import RouterConfig
from .efficiency import estimate_decoder_cost
from .models.llava import LlavaAdapter, MODEL_REGISTRY as LLAVA_MODELS
from .progress import configure_library_output
from .routing import route
from .types import RoutingContext


def _adapter(args: argparse.Namespace, config: RouterConfig):
    return LlavaAdapter(
        LLAVA_MODELS.get(args.model, args.model),
        config,
        args.device,
        args.dtype,
        args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
    )


def command_smoke(args: argparse.Namespace) -> None:
    generator = torch.Generator().manual_seed(args.seed)
    tokens = torch.randn(args.tokens, args.hidden, generator=generator)
    features = torch.randn(args.tokens, args.hidden, generator=generator)
    salience = torch.rand(args.tokens, generator=generator)
    text = torch.randn(args.text_tokens, args.hidden, generator=generator)
    side = int(args.tokens**0.5)
    grid = (side, side) if side * side == args.tokens else None
    config = RouterConfig(budget=args.budget, seed=args.seed)
    result = route(
        'stride',
        RoutingContext(
            tokens=tokens,
            vision_features=features,
            vision_salience=salience,
            text_tokens=text,
            grid=grid,
            salience_source='synthetic',
        ),
        config,
    )
    print(json.dumps({'shape': list(result.tokens.shape), **result.diagnostics}, indent=2))


def command_generate(args: argparse.Namespace) -> None:
    config = RouterConfig.from_yaml(args.config) if args.config else RouterConfig()
    result = _adapter(args, config).generate(
        args.image,
        args.prompt,
        method=args.method,
        budget=args.budget,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(result.__dict__, indent=2))


def command_evaluate(args: argparse.Namespace) -> None:
    from .evaluation.runner import evaluate_jsonl

    config = RouterConfig.from_yaml(args.config) if args.config else RouterConfig()
    summary = evaluate_jsonl(
        _adapter(args, config),
        args.data,
        args.output,
        args.method,
        args.budget,
        args.max_samples,
        not args.no_resume,
        {'max_new_tokens': args.max_new_tokens},
        config.seed,
    )
    print(json.dumps(summary, indent=2))


def command_estimate(args: argparse.Namespace) -> None:
    estimate = estimate_decoder_cost(
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        text_tokens=args.text_tokens,
        layers=args.layers,
        hidden_size=args.hidden_size,
        dtype_bytes=args.dtype_bytes,
    )
    print(json.dumps(estimate.to_dict(), indent=2))


def command_aggregate(args: argparse.Namespace) -> None:
    rows = aggregate_runs(args.root, args.csv)
    comparisons = paired_comparisons(
        args.root, args.paired_csv, reference=args.reference
    )
    write_latex_table(rows, args.tex)
    print(
        f'wrote {len(rows)} aggregate rows to {args.csv}, '
        f'{len(comparisons)} paired comparisons to {args.paired_csv}, '
        f'and table to {args.tex}'
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--model', default='llava15-7b')
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--dtype', default='bfloat16', choices=['float16', 'bfloat16', 'float32']
    )
    parser.add_argument('--attn-implementation', default=None)
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--method', default='stride')
    parser.add_argument('--budget', type=int, default=64)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--show-library-warnings', action='store_true')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='stride')
    sub = parser.add_subparsers(dest='command', required=True)
    smoke = sub.add_parser('smoke', help='run STRIDE without model downloads')
    smoke.add_argument('--tokens', type=int, default=576)
    smoke.add_argument('--text-tokens', type=int, default=16)
    smoke.add_argument('--hidden', type=int, default=128)
    smoke.add_argument('--budget', type=int, default=64)
    smoke.add_argument('--seed', type=int, default=0)
    smoke.set_defaults(func=command_smoke)

    generate = sub.add_parser('generate')
    add_model_args(generate)
    generate.add_argument('--image', type=Path, required=True)
    generate.add_argument('--prompt', required=True)
    generate.set_defaults(func=command_generate)

    evaluate = sub.add_parser('evaluate')
    add_model_args(evaluate)
    evaluate.add_argument('--data', type=Path, required=True)
    evaluate.add_argument('--output', type=Path, required=True)
    evaluate.add_argument('--max-samples', type=int)
    evaluate.add_argument('--no-resume', action='store_true')
    evaluate.set_defaults(func=command_evaluate)

    estimate = sub.add_parser('estimate')
    estimate.add_argument('--input-tokens', type=int, required=True)
    estimate.add_argument('--output-tokens', type=int, required=True)
    estimate.add_argument('--text-tokens', type=int, default=64)
    estimate.add_argument('--layers', type=int, required=True)
    estimate.add_argument('--hidden-size', type=int, required=True)
    estimate.add_argument('--dtype-bytes', type=int, default=2)
    estimate.set_defaults(func=command_estimate)

    aggregate = sub.add_parser('aggregate')
    aggregate.add_argument('--root', type=Path, default=Path('results/main'))
    aggregate.add_argument(
        '--csv', type=Path, default=Path('results/report/results.csv')
    )
    aggregate.add_argument(
        '--tex', type=Path, default=Path('results/report/results.tex')
    )
    aggregate.add_argument(
        '--paired-csv',
        type=Path,
        default=Path('results/report/paired_comparisons.csv'),
    )
    aggregate.add_argument('--reference', default='stride')
    aggregate.set_defaults(func=command_aggregate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_library_output(getattr(args, 'show_library_warnings', False))
    args.func(args)


if __name__ == '__main__':
    main()
