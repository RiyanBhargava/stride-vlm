import json

import pytest

from stride.config import RouterConfig
from stride.evaluation.runner import evaluate_jsonl, router_implementation_hash
from stride.models.base import GenerationOutput, VLMAdapter


class DummyAdapter(VLMAdapter):
    def __init__(self, config):
        self.router_config = config

    def generate(self, image, prompt, method, budget, **kwargs):
        return GenerationOutput(
            text='yes',
            input_visual_tokens=16,
            output_visual_tokens=budget,
            prefill_seconds=0.1,
            generation_seconds=0.1,
            peak_memory_bytes=None,
        )


def test_resume_rejects_changed_router_configuration(tmp_path):
    data = tmp_path / 'samples.jsonl'
    data.write_text(
        json.dumps(
            {
                'id': 'one',
                'image': 'unused.jpg',
                'question': 'Is it visible?',
                'answers': ['yes'],
                'metric': 'yes_no',
            }
        )
        + '\n',
        encoding='utf-8',
    )
    output = tmp_path / 'run'
    evaluate_jsonl(
        DummyAdapter(RouterConfig(budget=4)),
        data,
        output,
        'stride',
        4,
        1,
        True,
        {'max_new_tokens': 2},
    )
    with pytest.raises(ValueError, match='configuration changed'):
        evaluate_jsonl(
            DummyAdapter(
                RouterConfig(budget=4, stride_semantic_gain_min=0.2)
            ),
            data,
            output,
            'stride',
            4,
            1,
            True,
            {'max_new_tokens': 2},
        )


def test_sample_offset_selects_a_disjoint_slice(tmp_path):
    data = tmp_path / 'samples.jsonl'
    rows = [
        {
            'id': str(index), 'image': 'unused.jpg', 'question': 'Visible?',
            'answers': ['yes'], 'metric': 'yes_no',
        }
        for index in range(3)
    ]
    data.write_text(
        ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8'
    )
    output = tmp_path / 'offset_run'
    evaluate_jsonl(
        DummyAdapter(RouterConfig(budget=4)), data, output, 'stride', 4,
        max_samples=1, sample_offset=1,
    )
    record = json.loads((output / 'predictions.jsonl').read_text())
    assert record['id'] == '1'
    config = json.loads((output / 'run_config.json').read_text())
    assert config['sample_offset'] == 1


def test_router_implementation_hash_is_stable_and_recorded(tmp_path):
    first = router_implementation_hash()
    assert len(first) == 64
    assert first == router_implementation_hash()
    data = tmp_path / 'samples.jsonl'
    data.write_text(
        json.dumps(
            {
                'id': 'one', 'image': 'unused.jpg', 'question': 'Visible?',
                'answers': ['yes'], 'metric': 'yes_no',
            }
        ) + '\n',
        encoding='utf-8',
    )
    output = tmp_path / 'fingerprinted'
    evaluate_jsonl(
        DummyAdapter(RouterConfig(budget=4)), data, output, 'stride', 4, 1
    )
    run_config = json.loads((output / 'run_config.json').read_text())
    assert run_config['router_implementation_sha256'] == first
