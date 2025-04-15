import pytest

from mtb.layer_benchmarks.scaled_dot_product_attention import (
    ScaledDotProductAttentionBenchmark,
)
from tests.layer_benchmarks import LayerBenchmarkTest


class TestScaledDotProductAttentionBenchmark(LayerBenchmarkTest):
    @pytest.fixture
    def benchmark(self):
        return ScaledDotProductAttentionBenchmark(
            input_shape=(1, 3, 16),
            num_heads=8,
        )
