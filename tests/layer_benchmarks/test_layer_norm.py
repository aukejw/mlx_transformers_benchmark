import pytest

from mtb.layer_benchmarks.layer_norm import LayerNormBenchmark
from tests.layer_benchmarks import LayerBenchmarkTest


class TestLayerNormBenchmark(LayerBenchmarkTest):
    @pytest.fixture
    def benchmark(self):
        return LayerNormBenchmark(input_shape=(1, 3, 16))
