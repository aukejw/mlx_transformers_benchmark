import pytest

from mtb.benchmarks.layer_norm import LayerNormBenchmark
from mtb.benchmarks.linear import LinearBenchmark
from mtb.benchmarks.mhsa import MhsaBenchmark
from mtb.benchmarks.transformer_decoder_layer import TransformerDecoderLayerBenchmark
from mtb.benchmarks.transformer_encoder_layer import TransformerEncoderLayerBenchmark
from mtb.select_benchmarks import filter_benchmarks, layer_name_to_benchmark_class


@pytest.mark.parametrize(
    "layer_name, expected_class",
    [
        ("layernorm", LayerNormBenchmark),
        ("layer_norm", LayerNormBenchmark),
        ("linear", LinearBenchmark),
        ("mhsa", MhsaBenchmark),
        ("multiheadattention", MhsaBenchmark),
        ("transformerencoderlayer", TransformerEncoderLayerBenchmark),
        ("transformer_encoder_layer", TransformerEncoderLayerBenchmark),
        ("transformerdecoderlayer", TransformerDecoderLayerBenchmark),
        ("transformer_decoder_layer", TransformerDecoderLayerBenchmark),
    ],
)
def test_layer_name_to_benchmark_class(layer_name, expected_class):
    assert layer_name_to_benchmark_class(layer_name) == expected_class


def test_layer_name_to_benchmark_class_valueerror():
    with pytest.raises(ValueError, match="Unknown layer_name 'invalid_layer'"):
        layer_name_to_benchmark_class("invalid_layer")


def test_filter_benchmarks():
    input_shape = (1, 3, 16)
    benchmarks = [
        LayerNormBenchmark(input_shape=input_shape),
        LinearBenchmark(input_shape=input_shape),
        MhsaBenchmark(input_shape=input_shape),
        TransformerDecoderLayerBenchmark(input_shape=input_shape),
        TransformerEncoderLayerBenchmark(input_shape=input_shape),
    ]
    run_only_benchmarks = ["layernorm", "linear"]

    filtered_benchmarks = filter_benchmarks(benchmarks, run_only_benchmarks)

    assert len(filtered_benchmarks) == 2
    assert isinstance(filtered_benchmarks[0], LayerNormBenchmark)
    assert isinstance(filtered_benchmarks[1], LinearBenchmark)
