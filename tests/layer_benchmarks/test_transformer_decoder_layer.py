import pytest

from mtb import FLAG_ON_MAC
from mtb.layer_benchmarks.transformer_decoder_layer import (
    TransformerDecoderLayerBenchmark,
)
from tests.layer_benchmarks import LayerBenchmarkTest


class TestTransformerDecoderLayerBenchmark(LayerBenchmarkTest):
    @pytest.fixture
    def benchmark(self):
        return TransformerDecoderLayerBenchmark(
            feature_dim=16,
            num_heads=8,
            dropout=0.1,
        )

    @pytest.mark.skipif(not FLAG_ON_MAC, reason="Only works on Mac platform")
    def test_mlx_cpu(self, benchmark):
        self.benchmark_setup_run_teardown(benchmark, "mlx", "cpu")
