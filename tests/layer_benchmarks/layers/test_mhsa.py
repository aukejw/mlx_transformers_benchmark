import pytest

from mtb.layer_benchmarks.layers.mhsa import MhsaBenchmark
from tests.layer_benchmarks import LayerBenchmarkTest


class TestMhsaBenchmark(LayerBenchmarkTest):
    @pytest.fixture
    def benchmark(self):
        return MhsaBenchmark(
            feature_dim=16,
            num_heads=8,
        )
