import pytest

from mtb.layer_benchmarks.layers.scaled_dot_product_attention import (
    ScaledDotProductAttentionBenchmark,
)
from tests.layer_benchmarks import LayerBenchmarkTest


class TestScaledDotProductAttentionBenchmark(LayerBenchmarkTest):
    @pytest.fixture
    def benchmark(self):
        return ScaledDotProductAttentionBenchmark(
            feature_dim=16,
            num_heads=8,
            mask_type="causal",
        )
