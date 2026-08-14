#!/usr/bin/env python
"""Build paper-ready tables, comparisons, diagnostics, and figures."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from stride.analysis import aggregate_runs, paired_comparisons, write_latex_table

try:
    from scripts.summarize_results import build_report as build_markdown
except ImportError:
    from summarize_results import build_report as build_markdown


def diagnostics(root: Path) -> list[dict]:
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for path in root.rglob('predictions.jsonl'):
        relative = path.relative_to(root)
        if len(relative.parts) < 4:
            continue
        model, dataset = relative.parts[:2]
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                if row.get('method') != 'stride':
                    continue
                groups[(model, dataset, int(row['budget']))].append(
                    row.get('route_diagnostics', {})
                )
    rows = []
    for (model, dataset, budget), values in sorted(groups.items()):
        rows.append(
            {
                'model': model,
                'dataset': dataset,
                'budget': budget,
                'samples': len(values),
                'decoder_probe_layers_mean': mean(
                    float(value.get('decoder_probe_layers', 0))
                    for value in values
                ),
                'anchor_tokens_mean': mean(
                    float(value.get('anchor_tokens', 0)) for value in values
                ),
                'active_concepts_mean': mean(
                    float(value.get('active_concepts', 0)) for value in values
                ),
                'distribution_coverage_mean': mean(
                    float(value.get('distribution_coverage', 0))
                    for value in values
                ),
                'existence_intent_fraction': mean(
                    value.get('query_intent') == 'existence' for value in values
                ),
                'choice_intent_fraction': mean(
                    value.get('query_intent') == 'choice' for value in values
                ),
                'ocr_intent_fraction': mean(
                    value.get('query_intent') == 'ocr' for value in values
                ),
                'diversity_expert_fraction': mean(
                    value.get('selected_expert') == 'diversity' for value in values
                ),
                'residual_space_fraction': mean(
                    bool(value.get('residual_space_active', False))
                    for value in values
                ),
                'vision_space_fraction': mean(
                    bool(value.get('vision_space_active', False))
                    for value in values
                ),
                'semantic_active_fraction': mean(
                    bool(value.get('semantic_active', False)) for value in values
                ),
                'semantic_contrast_mean': mean(
                    float(value.get('semantic_contrast', 0)) for value in values
                ),
                'semantic_anchor_tokens_mean': mean(
                    float(value.get('semantic_anchor_tokens', 0))
                    for value in values
                ),
                'semantic_gain_mean': mean(
                    float(value.get('semantic_gain_total', 0)) for value in values
                ),
                'original_token_fraction': mean(
                    bool(value.get('emits_original_tokens', False))
                    for value in values
                ),
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _legacy_write_ablation_latex(csv_path: Path, output: Path) -> None:
    if not csv_path.exists():
        return
    with csv_path.open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    best = max(float(row['mean_score']) for row in rows)
    lines = [
        r'\begin{tabular}{lrrrr}',
        r'\toprule',
        r'Variant & Mean score & $\Delta$ vs. full & Lower & Sig. \\',
        r'\midrule',
    ]
    for row in rows:
        name = str(row['variant']).replace('_', r'\_')
        score = float(row['mean_score'])
        score_text = f'{100 * score:.2f}'
        if abs(score - best) < 1e-12:
            score_text = rf'\textbf{{{score_text}}}'
        lines.append(
            f"{name} & {score_text} & "
            f"{100 * float(row['mean_difference_vs_full']):+.2f} & "
            f"{row['cells_lower']} \\"
        )
    lines[4:] = [line + '\\' for line in lines[4:]]
    lines[6:] = [line + '\\' for line in lines[6:]]
    lines.extend((r'\bottomrule', r'\end{tabular}'))
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_journal_results_latex(rows: list[dict], output: Path) -> None:
    """Write a compact, matched-budget table for the manuscript."""
    # Keep this list aligned with configs/experiments/main.yaml. Pooling was
    # intentionally removed from the final comparison, so requiring it here
    # made a successful experiment fail only during report generation.
    methods = (
        'random', 'divprune', 'otprune',
        'visionzip', 'vispruner', 'stride',
    )
    dataset_labels = {
        'gqa': 'GQA',
        'pope': 'POPE',
        'scienceqa': 'ScienceQA',
        'textvqa': 'TextVQA',
    }
    lookup = {
        (str(row['dataset']), str(row['method']), int(row['budget'])): row
        for row in rows
    }
    lines = [
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{llrrrrrrr}',
        r'\toprule',
        (
            r'Budget & Dataset & Dense & Random & DivPrune & OTPrune & '
            r'VisionZip & VisPruner & \textbf{STRIDE} \\'
        ),
        r'\midrule',
    ]
    for budget_index, budget in enumerate((64, 128)):
        if budget_index:
            lines.append(r'\midrule')
        for dataset in ('gqa', 'pope', 'scienceqa', 'textvqa'):
            compressed = {
                method: float(lookup[(dataset, method, budget)]['score_mean'])
                for method in methods
            }
            best = max(compressed.values())
            dense = float(lookup[(dataset, 'dense', 576)]['score_mean'])
            values = []
            for method in methods:
                value = compressed[method]
                value_text = f'{100 * value:.2f}'
                if abs(value - best) < 1e-12:
                    value_text = rf'\textbf{{{value_text}}}'
                values.append(value_text)
            lines.append(
                f"{budget} & {dataset_labels[dataset]} & {100 * dense:.2f} & "
                + ' & '.join(values)
                + r' \\'
            )
    lines.extend((r'\bottomrule', r'\end{tabular}', r'}'))
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_journal_efficiency_latex(rows: list[dict], output: Path) -> None:
    """Write macro accuracy/latency trade-offs for the strongest controls."""
    selected = (
        'divprune', 'otprune', 'visionzip', 'vispruner', 'stride'
    )
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row['method']), int(row['budget']))].append(row)
    dense_rows = grouped[('dense', 576)]
    dense_score = mean(float(row['score_mean']) for row in dense_rows)
    dense_latency = mean(float(row['latency_mean_s']) for row in dense_rows)
    labels = {
        'divprune': 'DivPrune',
        'otprune': 'OTPrune',
        'visionzip': 'VisionZip',
        'vispruner': 'VisPruner',
        'stride': r'\textbf{STRIDE}',
    }
    lines = [
        r'\begin{tabular}{lrrrr}',
        r'\toprule',
        r'Method & Tokens & Reduction & Macro score & Speedup $\uparrow$ \\',
        r'\midrule',
        f'Dense & 576 & 0.0\\% & {100 * dense_score:.2f} & 1.00$\\times$ \\\\',
        r'\midrule',
    ]
    for budget_index, budget in enumerate((64, 128)):
        budget_rows = {method: grouped[(method, budget)] for method in selected}
        best_score = max(
            mean(float(row['score_mean']) for row in method_rows)
            for method_rows in budget_rows.values()
        )
        best_speedup = max(
            dense_latency / mean(float(row['latency_mean_s']) for row in method_rows)
            for method_rows in budget_rows.values()
        )
        if budget_index:
            lines.append(r'\addlinespace')
        for method in selected:
            method_rows = budget_rows[method]
            score = mean(float(row['score_mean']) for row in method_rows)
            latency = mean(float(row['latency_mean_s']) for row in method_rows)
            speedup = dense_latency / latency
            score_text = f'{100 * score:.2f}'
            speedup_text = f'{speedup:.2f}$\\times$'
            if abs(score - best_score) < 1e-12:
                score_text = rf'\textbf{{{score_text}}}'
            if abs(speedup - best_speedup) < 1e-12:
                speedup_text = rf'\textbf{{{speedup:.2f}$\times$}}'
            reduction = 100 * (1 - budget / 576)
            lines.append(
                f"{labels[method]} & {budget} & {reduction:.1f}\\% & "
                f"{score_text} & {speedup_text} \\"
            )
    lines.extend((r'\bottomrule', r'\end{tabular}'))
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_ablation_latex(csv_path: Path, output: Path) -> None:
    """Write the final component table with explicit significance counts."""
    if not csv_path.exists():
        return
    with csv_path.open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    best = max(float(row['mean_score']) for row in rows)
    row_end = '\\' * 2
    lines = [
        r'\begin{tabular}{lrrrr}',
        r'\toprule',
        f'Variant & Mean score & $\\Delta$ vs. full & Lower & Sig. {row_end}',
        r'\midrule',
    ]
    for row in rows:
        name = str(row['variant']).replace('_', r'\_')
        score = float(row['mean_score'])
        score_text = f'{100 * score:.2f}'
        if abs(score - best) < 1e-12:
            score_text = rf'\textbf{{{score_text}}}'
        lines.append(
            f"{name} & {score_text} & "
            f"{100 * float(row['mean_difference_vs_full']):+.2f} & "
            f"{row['cells_lower']} & {row['significant_lower']} {row_end}"
        )
    lines.extend((r'\bottomrule', r'\end{tabular}'))
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def repair_latex_row_endings(path: Path) -> None:
    """Defensively ensure every tabular row has a double backslash."""
    lines = path.read_text(encoding='utf-8').splitlines()
    repaired = []
    for line in lines:
        stripped = line.rstrip()
        if ' & ' in stripped and stripped.endswith('\\') and not stripped.endswith('\\\\'):
            stripped += '\\'
        repaired.append(stripped)
    path.write_text('\n'.join(repaired) + '\n', encoding='utf-8')


def write_supported_ablation_latex(csv_path: Path, output: Path) -> None:
    if not csv_path.exists():
        return
    with csv_path.open(encoding='utf-8') as handle:
        source = {row['variant']: row for row in csv.DictReader(handle)}
    names = [('full', 'None (full routed system)'),
             ('no_projected_geometry', 'Default projected geometry'),
             ('no_semantics', 'Semantic refinement'),
             ('no_modality_calibration', 'Modality calibration'),
             ('no_residual_space', 'Residual geometry')]
    end = '\\' * 2
    lines = [r'\begin{tabular}{lrrr}', r'\toprule',
             f'Removed component & Mean & $\\Delta$ & Lower {end}', r'\midrule']
    for variant, label in names:
        row = source[variant]
        score = '{:.2f}'.format(100 * float(row['mean_score']))
        delta = '{:+.2f}'.format(100 * float(row['mean_difference_vs_full']))
        lower = '{}/8'.format(row['cells_lower'])
        if variant == 'full':
            score, delta, lower = r'\textbf{{{}}}'.format(score), '--', '--'
        lines.append('{} & {} & {} & {} {}'.format(label, score, delta, lower, end))
    lines.extend((r'\bottomrule', r'\end{tabular}'))
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def ablation_markdown(csv_path: Path) -> str:
    if not csv_path.exists():
        return ''
    with csv_path.open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    labels = {
        'no_projected_geometry': 'Default projected geometry',
        'no_semantics': 'Semantic refinement',
        'no_modality_calibration': 'Modality calibration',
        'no_residual_space': 'Residual geometry',
        'no_intent_routing': 'Hard intent routing',
        'no_vision_space': 'OCR vision-space specialist',
        'no_diversity_expert': 'Existence diversity specialist',
    }
    lines = ['', '## Same-example component analysis', '',
             '> Supported removals and negative specialist findings are both retained.', '',
             '| Removed component | Mean | Difference vs full | Lower cells |',
             '|---|---:|---:|---:|']
    supported = [row for row in rows if row['variant'] != 'full'
                 and float(row['mean_difference_vs_full']) <= 0]
    for row in sorted(supported, key=lambda item: float(item['mean_difference_vs_full'])):
        lines.append('| {} | {:.2f} | {:+.2f} | {}/8 |'.format(
            labels.get(row['variant'], row['variant']), 100 * float(row['mean_score']),
            100 * float(row['mean_difference_vs_full']), row['cells_lower']))
    lines += ['', '### Unsupported specialist hypotheses', '',
              '| Removed component | Mean | Difference vs full |', '|---|---:|---:|']
    unsupported = [row for row in rows if float(row['mean_difference_vs_full']) > 0]
    for row in sorted(unsupported, key=lambda item: float(item['mean_difference_vs_full']), reverse=True):
        lines.append('| {} | {:.2f} | {:+.2f} |'.format(
            labels.get(row['variant'], row['variant']), 100 * float(row['mean_score']),
            100 * float(row['mean_difference_vs_full'])))
    lines += ['', 'No removal is significant after Holm correction. Positive differences mean removal scored higher; those branches are not claimed as beneficial.', '']
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('results/main'))
    parser.add_argument('--output', type=Path, default=Path('results/report'))
    parser.add_argument('--data-root', type=Path, default=Path('data_final'))
    parser.add_argument(
        '--ablation-root',
        type=Path,
        default=Path('results/ablations'),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results_csv = args.output / 'results.csv'
    paired_csv = args.output / 'paired_comparisons.csv'
    rows = aggregate_runs(args.root, results_csv)
    paired = paired_comparisons(args.root, paired_csv, reference='stride')
    if not rows:
        raise SystemExit(f'no completed predictions found under {args.root}')
    write_latex_table(rows, args.output / 'results.tex')
    ablation_csv = args.ablation_root / 'ablation_summary.csv'
    write_ablation_latex(ablation_csv, args.output / 'ablation_results.tex')
    write_supported_ablation_latex(
        ablation_csv, args.output / 'supported_ablation_results.tex'
    )
    diagnostic_rows = diagnostics(args.root)
    write_csv(diagnostic_rows, args.output / 'stride_diagnostics.csv')
    report = build_markdown(rows, paired, args.root, 'stride')
    report += ablation_markdown(ablation_csv)
    (args.output / 'results_report.md').write_text(report, encoding='utf-8')
    subprocess.run(
        [
            sys.executable,
            'scripts/plot_results.py',
            '--csv', str(results_csv),
            '--output', str(args.output / 'accuracy_efficiency.pdf'),
        ],
        check=True,
    )
    routing_inputs = [
        args.root / 'llava15-7b' / 'pope' / f'{method}_b64' / 'predictions.jsonl'
        for method in ('divprune', 'vispruner', 'visionzip', 'stride')
    ]
    if all(path.exists() for path in routing_inputs):
        subprocess.run(
            [
                sys.executable,
                'scripts/plot_routing_example.py',
                '--results-root', str(args.root),
                '--data-root', str(args.data_root),
                '--output', str(args.output / 'routing_selections.pdf'),
            ],
            check=True,
        )
    journal = Path('journal')
    if journal.exists():
        write_journal_results_latex(rows, journal / 'results.tex')
        write_journal_efficiency_latex(rows, journal / 'efficiency_results.tex')
        repair_latex_row_endings(journal / 'efficiency_results.tex')
        ablation_tex = args.output / 'ablation_results.tex'
        if ablation_tex.exists():
            shutil.copyfile(ablation_tex, journal / 'ablation_results.tex')
        supported_tex = args.output / 'supported_ablation_results.tex'
        if supported_tex.exists():
            shutil.copyfile(
                supported_tex, journal / 'supported_ablation_results.tex'
            )
    print(
        f'report complete: {len(rows)} aggregate rows, {len(paired)} paired '
        f'comparisons, {len(diagnostic_rows)} STRIDE diagnostic cells in {args.output}'
    )


if __name__ == '__main__':
    main()
