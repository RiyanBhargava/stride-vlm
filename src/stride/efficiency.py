from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CostEstimate:
    input_tokens: int
    output_tokens: int
    text_tokens: int
    layers: int
    hidden_size: int
    dense_prefill_flops: float
    routed_prefill_flops: float
    relative_prefill_flops: float
    dense_visual_kv_bytes: int
    routed_visual_kv_bytes: int
    relative_visual_kv: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def estimate_decoder_cost(
    input_tokens: int,
    output_tokens: int,
    text_tokens: int,
    layers: int,
    hidden_size: int,
    dtype_bytes: int = 2,
) -> CostEstimate:
    """Analytic decoder-only prefill/KV estimate.

    Uses 8*L*d^2 for attention projections + MLP and 4*L^2*d for attention.
    Constants cancel in ratios; measured latency remains the primary efficiency result.
    """
    dense_len, route_len = input_tokens + text_tokens, output_tokens + text_tokens

    def flops(seq: int) -> float:
        return float(layers * (8 * seq * hidden_size**2 + 4 * seq**2 * hidden_size))

    dense_flops, routed_flops = flops(dense_len), flops(route_len)
    dense_kv = 2 * layers * input_tokens * hidden_size * dtype_bytes
    route_kv = 2 * layers * output_tokens * hidden_size * dtype_bytes
    return CostEstimate(
        input_tokens,
        output_tokens,
        text_tokens,
        layers,
        hidden_size,
        dense_flops,
        routed_flops,
        routed_flops / dense_flops,
        dense_kv,
        route_kv,
        route_kv / dense_kv,
    )

