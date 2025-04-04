import pytest

from mtb import FLAG_ON_MAC
from mtb.benchmarks.transformer_encoder_layer import TransformerEncoderLayerBenchmark
from tests.benchmarks import BenchmarkTest


class TestTransformerEncoderLayerBenchmark(BenchmarkTest):
    @pytest.fixture
    def benchmark(self):
        return TransformerEncoderLayerBenchmark(
            input_shape=(1, 3, 16),
            num_heads=8,
            dropout=0.1,
        )

    @pytest.mark.skipif(not FLAG_ON_MAC, reason="Only works on Mac platform")
    def test_mlx_cpu(self, benchmark):
        self.benchmark_setup_run_teardown(benchmark, "mlx", "cpu")
