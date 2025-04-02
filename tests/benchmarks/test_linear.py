import pytest

from mtb.benchmarks.linear import LinearBenchmark
from tests.benchmarks import BenchmarkTest


class TestLinearBenchmark(BenchmarkTest):
    @pytest.fixture
    def benchmark(self):
        return LinearBenchmark(
            input_shapes=[(1, 3, 16)],
        )
