import pytest

from mtb.layer_benchmarks.softmax import SoftmaxBenchmark
from tests.layer_benchmarks import LayerBenchmarkTest


class TestSoftmax(LayerBenchmarkTest):
    @pytest.fixture
    def benchmark(self):
        return SoftmaxBenchmark(feature_dim=16)
