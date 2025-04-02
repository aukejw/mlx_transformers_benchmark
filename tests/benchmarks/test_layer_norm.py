import pytest

from mtb.benchmarks.layer_norm import LayerNormBenchmark
from tests.benchmarks import BenchmarkTest


class TestLayerNormBenchmark(BenchmarkTest):
    @pytest.fixture
    def benchmark(self):
        return LayerNormBenchmark(input_shape=(1, 3, 16))
