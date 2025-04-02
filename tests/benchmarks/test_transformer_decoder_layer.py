import pytest

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
