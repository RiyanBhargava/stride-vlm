#!/usr/bin/env python
"""Structural STRIDE ablations for the exact-reference study."""

from __future__ import annotations


STRIDE_ABLATIONS: dict[str, dict[str, object]] = {
    'no_semantics': {
        'stride_use_semantics': False,
    },
    'no_intent_routing': {
        'stride_use_intent_routing': False,
    },
    'no_diversity_expert': {
        'stride_use_diversity_expert': False,
    },
    'no_residual_space': {
        'stride_use_residual_space': False,
    },
    'no_projected_geometry': {
        'stride_use_projected_geometry': False,
    },
    'no_vision_space': {
        'stride_use_vision_space': False,
    },
    'no_modality_calibration': {
        'stride_use_modality_calibration': False,
    },
}


if __name__ == '__main__':
    names = ', '.join(STRIDE_ABLATIONS)
    raise SystemExit(
        'Use scripts/run_full_ablations.py --config '
        'configs/experiments/ablations.yaml. Available variants: ' + names
    )
