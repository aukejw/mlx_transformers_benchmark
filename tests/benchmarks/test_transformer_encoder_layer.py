import pytest

from mtb.benchmarks.transformer_encoder_layer import TransformerEncoderLayerBenchmark
from tests.benchmarks import BenchmarkTest


class TestTransformerEncoderLayerBenchmark(BenchmarkTest):
    @pytest.fixture
    def benchmark(self):
        return TransformerEncoderLayerBenchmark(
            input_shapes=[(1, 3, 16)],
            num_heads=8,
            dropout=0.1,
        )
