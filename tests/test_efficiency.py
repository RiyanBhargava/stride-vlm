from stride.efficiency import estimate_decoder_cost


def test_cost_reduces_monotonically():
    estimate = estimate_decoder_cost(576, 64, 40, 32, 4096)
    assert 0 < estimate.relative_prefill_flops < 1
    assert estimate.relative_visual_kv == 64 / 576

