import importlib
from typing import List, Type

from mtb.benchmarks.base_benchmark import BaseBenchmark


def benchmark_name_to_benchmark_class(
    benchmark_name: str,
) -> Type:
    """Get a benchmark class from a vague string identifier.

    Args:
        benchmark_name: String identifier for a benchmark.

    """
    original_benchmark_name = benchmark_name
    benchmark_name = benchmark_name.lower().replace("_", "")

    from mtb.benchmarks import __all__ as operator_benchmark_names
    from mtb.llm_benchmarks import __all__ as llm_benchmark_names

    name_to_benchmark_class = dict()
    for name in operator_benchmark_names:
        name_to_benchmark_class[name.lower()] = f"mtb.benchmarks.{name}"
    for name in llm_benchmark_names:
        name_to_benchmark_class[name.lower()] = f"mtb.llm_benchmarks.{name}"

    if not benchmark_name.endswith("benchmark"):
        benchmark_name += "benchmark"

    try:
        benchmark_class_name = name_to_benchmark_class[benchmark_name]
    except KeyError:
        raise ValueError(
            f"Could not find benchmark class for name '{original_benchmark_name}'"
        )

    module_name, class_name = benchmark_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_type = getattr(module, class_name)

    return class_type


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
        benchmark_name_to_benchmark_class(name) for name in run_only_benchmarks
    )
    filtered_benchmarks = [
        benchmark
        for benchmark in benchmarks
        if benchmark.__class__ in valid_benchmark_classes
    ]
    return filtered_benchmarks
