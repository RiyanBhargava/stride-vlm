#!/usr/bin/env python
'''Visualize which image patches each selector retains for one shared example.'''

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def load_predictions(path: Path) -> dict[str, dict]:
    with path.open('r', encoding='utf-8') as handle:
        return {
            str(row['id']): row
            for row in (json.loads(line) for line in handle if line.strip())
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', type=Path, default=Path('results/main'))
    parser.add_argument('--data-root', type=Path, default=Path('data_final'))
    parser.add_argument('--model', default='llava15-7b')
    parser.add_argument('--dataset', default='pope')
    parser.add_argument('--budget', type=int, default=64)
    parser.add_argument('--sample-id')
    parser.add_argument(
        '--output', type=Path, default=Path('results/report/routing_selections.pdf')
    )
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit('install plotting dependencies with pip install -e .[eval]') from error

    methods = ('divprune', 'vispruner', 'visionzip', 'stride')
    predictions = {
        method: load_predictions(
            args.results_root
            / args.model
            / args.dataset
            / f'{method}_b{args.budget}'
            / 'predictions.jsonl'
        )
        for method in methods
    }
    shared = set.intersection(*(set(rows) for rows in predictions.values()))
    sample_id = args.sample_id or (sorted(shared)[0] if shared else None)
    if sample_id is None or sample_id not in shared:
        raise ValueError('no shared sample ID exists across routing methods')

    data_path = args.data_root / args.dataset / 'samples.jsonl'
    with data_path.open('r', encoding='utf-8') as handle:
        examples = {
            str(row['id']): row
            for row in (json.loads(line) for line in handle if line.strip())
        }
    example = examples[sample_id]
    image_path = Path(example['image'])
    if not image_path.is_absolute():
        image_path = data_path.parent / image_path
    image = Image.open(image_path).convert('RGB')

    figure, axes = plt.subplots(1, len(methods), figsize=(12, 3.3), constrained_layout=True)
    for axis, method in zip(axes, methods):
        row = predictions[method][sample_id]
        diagnostics = row.get('route_diagnostics', {})
        selected = diagnostics.get('selected_indices')
        if selected is None:
            raise ValueError(f'{method} predictions do not contain selected indices')
        grid = diagnostics.get('input_grid')
        if grid is None:
            side = round(math.sqrt(int(row['input_visual_tokens'])))
            grid = [side, side]
        height, width = map(int, grid)
        mask = np.zeros((height, width), dtype=float)
        flat = mask.reshape(-1)
        patch_indices = np.asarray(selected, dtype=int)
        patch_indices = patch_indices[patch_indices >= 0]
        flat[patch_indices] = 1.0
        axis.imshow(image)
        axis.imshow(
            mask,
            cmap='Reds',
            alpha=0.45 * mask,
            interpolation='nearest',
            extent=(0, image.width, image.height, 0),
            vmin=0,
            vmax=1,
        )
        axis.set_title(f'{method} ({len(selected)} tokens)')
        axis.axis('off')
    question = str(example['question'])
    figure.suptitle(f'{args.dataset.upper()} sample {sample_id}: {question}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    print(f'wrote routing visualization to {args.output}')


if __name__ == '__main__':
    main()
