import pytest

from mtb.benchmarks.mhsa import MhsaBenchmark
from tests.benchmarks import BenchmarkTest


class TestMultiheadAttentionBenchmark(BenchmarkTest):
    @pytest.fixture
    def benchmark(self):
        return MhsaBenchmark(
            input_shapes=[(1, 3, 16)],
            num_heads=8,
        )
