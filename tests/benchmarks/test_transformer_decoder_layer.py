import pytest

from mtb import FLAG_ON_MAC
from mtb.benchmarks.transformer_decoder_layer import TransformerDecoderLayerBenchmark
from tests.benchmarks import BenchmarkTest


class TestTransformerDecoderLayerBenchmark(BenchmarkTest):
    @pytest.fixture
    def benchmark(self):
        return TransformerDecoderLayerBenchmark(
            input_shape=(1, 3, 16),
            num_heads=8,
            dropout=0.1,
        )

    @pytest.mark.skipif(not FLAG_ON_MAC, reason="Only works on Mac platform")
    def test_mlx_cpu(self, benchmark):
        self.benchmark_setup_run_teardown(benchmark, "mlx", "cpu")
