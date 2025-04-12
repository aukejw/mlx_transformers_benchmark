import pytest

from mtb.benchmarks.softmax import SoftmaxBenchmark
from tests.benchmarks import BenchmarkTest


class TestSoftmax(BenchmarkTest):
    @pytest.fixture
    def benchmark(self):
        return SoftmaxBenchmark(
            input_shape=(1, 3, 16),
        )
