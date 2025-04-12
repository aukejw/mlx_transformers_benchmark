from typing import List, Type

from mtb.benchmarks import (
    LayerNormBenchmark,
    LinearBenchmark,
    MhsaBenchmark,
    ScaledDotProductAttentionBenchmark,
    SoftmaxBenchmark,
    TransformerDecoderLayerBenchmark,
    TransformerEncoderLayerBenchmark,
)
from mtb.benchmarks.base_benchmark import BaseBenchmark


def layer_name_to_benchmark_class(
    layer_name: str,
) -> Type:
    """Get a benchmark class from a vague string identifier.

    Args:
        layer_name: String identifier for a layer class.

    """
    layer_name = layer_name.lower()
    if layer_name in ("layernorm", "layer_norm"):
        return LayerNormBenchmark
    elif layer_name in ("linear",):
        return LinearBenchmark
    elif layer_name in ("mhsa", "multiheadattention"):
        return MhsaBenchmark
    elif layer_name in ("softmax"):
        return SoftmaxBenchmark
    elif layer_name in ("scaled_dot_product_attention", "sdpa"):
        return ScaledDotProductAttentionBenchmark
    elif layer_name in ("transformerencoderlayer", "transformer_encoder_layer"):
        return TransformerEncoderLayerBenchmark
    elif layer_name in ("transformerdecoderlayer", "transformer_decoder_layer"):
        return TransformerDecoderLayerBenchmark
    else:
        raise ValueError(f"Unknown layer_name '{layer_name}'")


def filter_benchmarks(
    benchmarks: List[BaseBenchmark],
    run_only_benchmarks: List[str],
) -> List[BaseBenchmark]:
    """Filter given benchmarks, return only the ones meeting the criterion.

    Args:
        benchmarks: List of benchmarks to filter.
        run_only_benchmarks: List of benchmark names to include.

    """
    valid_benchmark_classes = set(
        layer_name_to_benchmark_class(name) for name in run_only_benchmarks
    )
    filtered_benchmarks = [
        benchmark
        for benchmark in benchmarks
        if benchmark.__class__ in valid_benchmark_classes
    ]
    return filtered_benchmarks
