import pytest

from mtb.layer_benchmarks.mhsa import MhsaBenchmark
from tests.layer_benchmarks import LayerBenchmarkTest


class TestMhsaBenchmark(LayerBenchmarkTest):
    @pytest.fixture
    def benchmark(self):
        return MhsaBenchmark(
            input_shape=(1, 3, 16),
            num_heads=8,
        )
