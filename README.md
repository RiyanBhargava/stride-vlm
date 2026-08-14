# STRIDE-VLM

Research code for **Training-Free Semantic Token Routing in VLMs: Closing the Modality–Positional Gap at Inference**.

STRIDE reduces LLaVA-1.5 from 576 visual tokens to 64 or 128 without training. It combines frozen visual salience, projected VLM geometry, shared cross-modal residual evidence, calibrated question concepts, and raster-order output packing. Selected outputs are original projected visual tokens. No model weight, label, learned router, answer ensemble, or decoder-attention probe is used.

## Final audited results

The final study uses 200 disjoint examples each from POPE, TextVQA, ScienceQA, and GQA. Every task has zero overlap with 964 previously viewed IDs.

| Method | 64 tokens | 128 tokens |
|---|---:|---:|
| Dense, 576 tokens | 66.76% | 66.76% |
| Random, three seeds | 57.50% | 60.38% |
| DivPrune | 61.82% | 63.41% |
| OTPrune | 62.34% | 64.03% |
| VisionZip | 61.42% | 62.34% |
| VisPruner | 61.79% | 64.92% |
| **STRIDE** | **62.55%** | **65.72%** |

Within this declared comparison set, STRIDE has the highest measured macro score at both budgets:

- **64 tokens:** 62.55%, a **+0.21-point** measured lead, 88.9% token reduction, and 1.31x speedup.
- **128 tokens:** 65.72%, a **+0.80-point** measured lead, 77.8% token reduction, and 1.25x speedup.

The 128-token setting wins GQA, ScienceQA, and TextVQA among compressed methods and retains approximately 98.4% of dense macro performance.

These are point-estimate leads. They do not establish global state of the art or corrected significance against the strongest method in every individual benchmark-budget cell.

## Component result

Same-example ablations show that projected geometry and calibrated semantics are the best-supported pieces. Removing projected geometry lowers the aggregate by 0.69 points; removing semantic refinement or modality calibration lowers it by 0.24 points. Hard intent routing and two specialized branches do not improve the final aggregate and are reported as experimental limitations.

## Install

Python 3.10 or newer and an NVIDIA GPU are recommended.

```powershell
python -m pip install -e '.[all]'
```

The supplied configuration uses cached LLaVA-1.5-7B weights with four-bit loading.

## Reproduce everything

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_final_pipeline.ps1
```

The pipeline:

1. runs all unit tests;
2. prepares and validates the disjoint data split;
3. runs Dense, Random with three seeds, DivPrune, OTPrune, VisPruner, VisionZip, and STRIDE;
4. runs seven same-data STRIDE ablations at 64 and 128 tokens;
5. independently recomputes every score; and
6. generates Markdown, CSV, LaTeX, statistical, diagnostic, and PDF outputs.

Every long stage displays elapsed time, processing rate, and ETA. Compatible completed predictions are reused safely.

## Generated outputs

| Path | Meaning |
|---|---|
| `results/main/` | main predictions and summaries |
| `results/ablations/` | exact same-data component removals |
| `results/report/results_report.md` | readable final report |
| `results/report/results.csv` | aggregate accuracy and timing |
| `results/report/paired_comparisons.csv` | paired uncertainty and corrected tests |
| `results/report/stride_diagnostics.csv` | router activation diagnostics |

The complete audited `results/` tree is versioned publicly. The `docs/` and
`journal/` folders remain local-only and excluded from Git tracking.

## Reproducibility

- Random is averaged over three seeds; all other routers and greedy generation are deterministic.
- Every prediction records its router configuration, code state, evaluation version, timings, selected-token diagnostics, and environment metadata.
- Score auditing recomputes prediction rows and summaries independently.
- Resume validation refuses to mix incompatible router or generation configurations.
- The final evaluation IDs must not be used for additional tuning.

## Claim boundary

The supported claim is that STRIDE has the highest measured macro average among the declared comparison set at 64 and 128 tokens on this disjoint study. The strongest result is the 128-token accuracy-efficiency trade-off. The ablation does not support claiming that every intent-specific branch independently improves accuracy.

## Novelty scope as of August 2026

STRIDE does not claim to invent semantic pruning, structure--semantic staging,
missing-evidence selection, or cross-modal residualization individually. Recent
work such as [DIVE](https://arxiv.org/abs/2608.04496),
[STS](https://arxiv.org/abs/2606.03569), and
[SIEVE](https://arxiv.org/abs/2608.10489) overlaps those principles.

The potentially distinct contribution is the audited interface combination of
a shared frozen first-layer value map, robust within-image concept calibration,
bounded answer-independent replacement, and raster-ordered emission of unchanged
projected tokens. This is an incremental system contribution until it is tested
faithfully against those concurrent methods on a new untouched split.
