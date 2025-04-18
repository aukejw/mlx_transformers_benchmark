import pytest

from mtb.layer_benchmarks.linear import LinearBenchmark
from tests.layer_benchmarks import LayerBenchmarkTest


class TestLinearBenchmark(LayerBenchmarkTest):
    @pytest.fixture
    def benchmark(self):
        return LinearBenchmark(feature_dim=16)
