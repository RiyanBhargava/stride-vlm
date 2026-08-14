# Current measured results

> Generated from raw prediction JSONL and aggregate CSV files. Do not edit numerical values by hand.

## Completeness

- Prediction files: 68
- Prediction rows: 13,600
- Aggregate groups: 52
- Paired comparisons: 48

## Macro-average by model

| Model | Method | Budget | Mean score | Mean latency (ms) |
|---|---|---:|---:|---:|
| llava15-7b | dense | 576 | 0.668 | 356.9 |
| llava15-7b | divprune | 64 | 0.618 | 246.8 |
| llava15-7b | divprune | 128 | 0.634 | 286.8 |
| llava15-7b | otprune | 64 | 0.623 | 264.1 |
| llava15-7b | otprune | 128 | 0.640 | 326.0 |
| llava15-7b | random | 64 | 0.575 | 238.2 |
| llava15-7b | random | 128 | 0.604 | 244.4 |
| llava15-7b | stride | 64 | 0.626 | 272.6 |
| llava15-7b | stride | 128 | 0.657 | 286.2 |
| llava15-7b | visionzip | 64 | 0.614 | 266.7 |
| llava15-7b | visionzip | 128 | 0.623 | 308.8 |
| llava15-7b | vispruner | 64 | 0.618 | 267.5 |
| llava15-7b | vispruner | 128 | 0.649 | 282.8 |

## stride against dense and the strongest compressed baseline

| Model | Dataset | Budget | Reference | Dense | Best compressed control | Ref. - best | Token reduction | Speedup vs dense |
|---|---|---:|---:|---:|---|---:|---:|---:|
| llava15-7b | gqa | 64 | 0.640 | 0.755 | otprune (0.680) | -0.040 | 88.9% | 1.36x |
| llava15-7b | gqa | 128 | 0.700 | 0.755 | vispruner (0.690) | 0.010 | 77.8% | 1.30x |
| llava15-7b | pope | 64 | 0.795 | 0.805 | otprune (0.815) | -0.020 | 88.9% | 1.22x |
| llava15-7b | pope | 128 | 0.830 | 0.805 | otprune (0.835) | -0.005 | 77.8% | 1.14x |
| llava15-7b | scienceqa | 64 | 0.600 | 0.590 | divprune (0.610) | -0.010 | 88.9% | 1.30x |
| llava15-7b | scienceqa | 128 | 0.605 | 0.590 | vispruner (0.595) | 0.010 | 77.8% | 1.30x |
| llava15-7b | textvqa | 64 | 0.467 | 0.520 | vispruner (0.452) | 0.016 | 88.9% | 1.34x |
| llava15-7b | textvqa | 128 | 0.494 | 0.520 | vispruner (0.492) | 0.002 | 77.8% | 1.25x |

## Multiplicity-corrected paired findings

- Significant positive comparisons: 3
- Significant negative comparisons: 1

| Direction | Model | Dataset | Budget | Comparison | Difference | 95% CI | Holm p |
|---|---|---|---:|---|---:|---|---:|
| negative | llava15-7b | gqa | 64 | dense | -0.115 | [-0.170, -0.065] | 0.0006 |
| positive | llava15-7b | textvqa | 64 | otprune | 0.073 | [0.021, 0.126] | 0.0405 |
| positive | llava15-7b | textvqa | 64 | random | 0.196 | [0.143, 0.249] | 0.0006 |
| positive | llava15-7b | textvqa | 128 | random | 0.136 | [0.085, 0.186] | 0.0006 |

## Interpretation rule

A new router is not considered supported merely because it compresses tokens. It must beat the strongest matched-budget control with paired uncertainty, while retaining a measured latency advantage. Any method changed after inspecting final outcomes requires a new disjoint evaluation set.

## Same-example component analysis

> Supported removals and negative specialist findings are both retained.

| Removed component | Mean | Difference vs full | Lower cells |
|---|---:|---:|---:|
| Default projected geometry | 63.44 | -0.69 | 6/8 |
| Modality calibration | 63.89 | -0.24 | 2/8 |
| Semantic refinement | 63.89 | -0.24 | 2/8 |
| Residual geometry | 64.08 | -0.06 | 1/8 |

### Unsupported specialist hypotheses

| Removed component | Mean | Difference vs full |
|---|---:|---:|
| Hard intent routing | 64.72 | +0.59 |
| OCR vision-space specialist | 64.54 | +0.40 |
| Existence diversity specialist | 64.45 | +0.31 |

No removal is significant after Holm correction. Positive differences mean removal scored higher; those branches are not claimed as beneficial.
