import pytest

from mtb.layer_benchmarks.layer_norm import LayerNormBenchmark
from mtb.layer_benchmarks.linear import LinearBenchmark
from mtb.layer_benchmarks.mhsa import MhsaBenchmark
from mtb.layer_benchmarks.scaled_dot_product_attention import (
    ScaledDotProductAttentionBenchmark,
)
from mtb.layer_benchmarks.transformer_decoder_layer import (
    TransformerDecoderLayerBenchmark,
)
from mtb.layer_benchmarks.transformer_encoder_layer import (
    TransformerEncoderLayerBenchmark,
)
from mtb.select_benchmarks import benchmark_name_to_benchmark_class, filter_benchmarks


@pytest.mark.parametrize(
    "benchmark_name, expected_class",
    [
        ("layernorm", LayerNormBenchmark),
        ("linear", LinearBenchmark),
        ("mhsa", MhsaBenchmark),
        ("scaled_dot_product_attention", ScaledDotProductAttentionBenchmark),
        ("transformerencoderlayer", TransformerEncoderLayerBenchmark),
        ("transformerdecoderlayer", TransformerDecoderLayerBenchmark),
    ],
)
def test_benchmark_name_to_benchmark_class(benchmark_name, expected_class):
    assert benchmark_name_to_benchmark_class(benchmark_name) == expected_class


def test_benchmark_name_to_benchmark_class_valueerror():
    with pytest.raises(
        ValueError,
        match="Could not find benchmark class for name 'invalid_benchmark'",
    ):
        benchmark_name_to_benchmark_class("invalid_benchmark")


@pytest.fixture()
def layer_benchmarks():
    kwargs = dict(
        feature_dim=16,
    )
    return [
        LayerNormBenchmark(**kwargs),
        LinearBenchmark(**kwargs),
        MhsaBenchmark(**kwargs),
        TransformerDecoderLayerBenchmark(**kwargs),
        TransformerEncoderLayerBenchmark(**kwargs),
    ]


def test_filter_benchmarks(layer_benchmarks):
    run_only_benchmarks = ["layernorm", "linear"]
    filtered_benchmarks = filter_benchmarks(layer_benchmarks, run_only_benchmarks)
    assert len(filtered_benchmarks) == 2
    assert isinstance(filtered_benchmarks[0], LayerNormBenchmark)
    assert isinstance(filtered_benchmarks[1], LinearBenchmark)


def test_filter_benchmarks_str(layer_benchmarks):
    run_only_benchmarks = "layernorm"
    filtered_benchmarks = filter_benchmarks(layer_benchmarks, run_only_benchmarks)
    assert len(filtered_benchmarks) == 1
    assert isinstance(filtered_benchmarks[0], LayerNormBenchmark)
