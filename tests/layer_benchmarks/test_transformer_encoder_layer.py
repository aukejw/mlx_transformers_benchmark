import pytest

from mtb import FLAG_ON_MAC
from mtb.layer_benchmarks.transformer_encoder_layer import (
    TransformerEncoderLayerBenchmark,
)
from tests.layer_benchmarks import LayerBenchmarkTest


class TestTransformerEncoderLayerBenchmark(LayerBenchmarkTest):
    @pytest.fixture
    def benchmark(self):
        return TransformerEncoderLayerBenchmark(
            feature_dim=16,
            num_heads=8,
            dropout=0.1,
        )

    @pytest.mark.skipif(not FLAG_ON_MAC, reason="Only works on Mac platform")
    def test_mlx_cpu(self, benchmark):
        self.benchmark_setup_run_teardown(benchmark, "mlx", "cpu")
