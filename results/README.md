# Published result artifacts

This directory contains the exact final artifacts used by the STRIDE report.

| Path | Contents |
|---|---|
| `main/` | 13,600 predictions from 68 main runs |
| `ablations/` | 11,200 predictions from 56 same-example component removals |
| `report/results_report.md` | readable measured-result summary |
| `report/results.csv` | aggregate scores, token counts, memory, and latency |
| `report/paired_comparisons.csv` | paired intervals, tests, and Holm correction |
| `report/ablation_results.tex` | complete aggregate ablation record |
| `report/supported_ablation_results.tex` | compact supported-core view |
| `report/complete_ablation_summary.csv` | unfiltered aggregate ablation CSV |
| `report/supported_ablation_summary.csv` | full system plus score-lowering removals |
| `report/unsupported_specialist_summary.csv` | removals that scored above the full system |
| `report/stride_diagnostics.csv` | observed branch activations |
| `report/*.pdf` | generated accuracy/efficiency and routing figures |

All 24,800 prediction rows recompute exactly with evaluator version
`evalai-vqa-v1`. The compact ablation view does not delete negative findings:
specialist removals that score above full STRIDE remain in the complete table,
CSV files, raw predictions, and Markdown report.

Regenerate the reports from the repository root with:

```powershell
python scripts/build_report.py --root results/main --output results/report --data-root data_final --ablation-root results/ablations
```

The image dataset itself is not included.
