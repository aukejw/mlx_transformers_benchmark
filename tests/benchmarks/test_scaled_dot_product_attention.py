import pytest

from mtb.benchmarks.scaled_dot_product_attention import (
    ScaledDotProductAttentionBenchmark,
)
from tests.benchmarks import BenchmarkTest


class TestScaledDotProductAttentionBenchmark(BenchmarkTest):
    @pytest.fixture
    def benchmark(self):
        return ScaledDotProductAttentionBenchmark(
            input_shape=(1, 3, 16),
            num_heads=8,
        )
