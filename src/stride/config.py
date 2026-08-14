from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RouterConfig:
    """Configuration shared by the official baselines and STRIDE.

    STRIDE's switches correspond to distinct selection operations.  The full
    ablation smoke test verifies that every declared component changes the
    selected token set on real model inputs.
    """

    method: str = 'stride'
    budget: int = 64
    vispruner_important_ratio: float = 0.5
    visionzip_context_tokens: int = 30
    otprune_gamma: float = 0.01
    stride_concept_contrast_min: float = 1.0
    stride_semantic_anchor_fraction: float = 0.0625
    stride_semantic_gain_min: float = 0.15
    stride_residual_ridge: float = 0.1
    stride_use_semantics: bool = True
    stride_use_modality_calibration: bool = True
    stride_use_residual_space: bool = True
    stride_use_projected_geometry: bool = True
    stride_use_vision_space: bool = True
    stride_use_intent_routing: bool = True
    stride_use_diversity_expert: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.method != 'stride':
            raise ValueError('method must be stride')
        if self.budget < 1:
            raise ValueError('budget must be positive')
        if self.visionzip_context_tokens < 1:
            raise ValueError('visionzip_context_tokens must be positive')
        for name in (
            'vispruner_important_ratio',
            'stride_semantic_anchor_fraction',
            'stride_semantic_gain_min',
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f'{name} must be in [0, 1]')
        if self.stride_concept_contrast_min < 0:
            raise ValueError('stride_concept_contrast_min must be non-negative')
        if self.stride_residual_ridge <= 0:
            raise ValueError('stride_residual_ridge must be positive')
        for name in (
            'otprune_gamma',
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> 'RouterConfig':
        allowed = {field.name for field in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f'Unknown router keys: {sorted(unknown)}')
        return cls(**raw)

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'RouterConfig':
        with Path(path).open('r', encoding='utf-8') as handle:
            raw = yaml.safe_load(handle) or {}
        if 'router' in raw:
            raw = raw['router']
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
