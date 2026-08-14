import json

import pytest

from scripts.run_full_ablations import summarize, validate_full_reference
from stride.config import RouterConfig


def write_reference(tmp_path, ids=('a', 'b'), budget=8):
    data = tmp_path / 'samples.jsonl'
    data.write_text(
        ''.join(
            json.dumps(
                {
                    'id': item,
                    'image': f'{item}.jpg',
                    'question': 'Question?',
                    'answers': ['answer'],
                }
            )
            + '\n'
            for item in ids
        ),
        encoding='utf-8',
    )
    predictions = tmp_path / 'predictions.jsonl'
    predictions.write_text(
        ''.join(
            json.dumps(
                {'id': item, 'method': 'stride', 'budget': budget, 'score': score}
            )
            + '\n'
            for item, score in zip(ids, (1.0, 0.0))
        ),
        encoding='utf-8',
    )
    config = RouterConfig(budget=budget)
    summary_path = tmp_path / 'summary.json'
    summary_path.write_text(
        json.dumps(
            {
                'method': 'stride',
                'budget': budget,
                'samples': len(ids),
                'router_config': config.to_dict(),
                'generation_kwargs': {'max_new_tokens': 12},
            }
        ),
        encoding='utf-8',
    )
    return data, predictions, summary_path, config


def test_validate_full_reference_accepts_exact_main_run(tmp_path):
    data, predictions, summary_path, config = write_reference(tmp_path)
    result = validate_full_reference(
        data, predictions, summary_path, 8, 2, config, 12
    )
    assert result['samples'] == 2
    assert result['score'] == 0.5
    assert len(result['sha256']) == 64


def test_validate_full_reference_rejects_changed_order(tmp_path):
    data, predictions, summary_path, config = write_reference(tmp_path)
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    predictions.write_text(
        ''.join(json.dumps(row) + '\n' for row in reversed(rows))
    )
    with pytest.raises(ValueError, match='does not exactly match'):
        validate_full_reference(
            data, predictions, summary_path, 8, 2, config, 12
        )


def test_summarize_counts_directions():
    rows = [
        {
            'variant': 'no_semantics',
            'score_mean': 0.7,
            'difference_vs_full': -0.1,
            'p_holm': 0.04,
        },
        {
            'variant': 'no_semantics',
            'score_mean': 0.9,
            'difference_vs_full': 0.1,
            'p_holm': 0.20,
        },
    ]
    result = summarize(rows)[0]
    assert result['mean_difference_vs_full'] == pytest.approx(0.0)
    assert result['cells_higher'] == 1
    assert result['cells_lower'] == 1
    assert result['significant_lower'] == 1
