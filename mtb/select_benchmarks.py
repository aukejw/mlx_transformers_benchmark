import importlib
from typing import List, Type, Union

from mtb.layer_benchmarks.base_layer_benchmark import BaseLayerBenchmark


def benchmark_name_to_benchmark_class(
    benchmark_name: str,
) -> Type:
    """Get a benchmark class from a vague string identifier.

    Args:
        benchmark_name: String identifier for a benchmark.

    """
    original_benchmark_name = benchmark_name
    benchmark_name = benchmark_name.lower().replace("_", "")

    from mtb.layer_benchmarks import __all__ as layer_benchmark_names
    from mtb.llm_benchmarks import __all__ as llm_benchmark_names

    name_to_benchmark_class = dict()
    for name in layer_benchmark_names:
        name_to_benchmark_class[name.lower()] = f"mtb.layer_benchmarks.{name}"
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
    benchmarks: List[BaseLayerBenchmark],
    run_only_benchmarks: Union[str, List[str]],
) -> List[BaseLayerBenchmark]:
    """Filter given benchmarks, return only the ones meeting the criterion.

    Args:
        benchmarks: List of benchmarks to filter.
        run_only_benchmarks: List of benchmark names to include.

    """
    if isinstance(run_only_benchmarks, str):
        run_only_benchmarks = [run_only_benchmarks]

    run_only_benchmarks = [
        benchmark_name.lower().replace("_", "")
        for benchmark_name in run_only_benchmarks
    ]

    benchmarks_to_keep = []
    for benchmark in benchmarks:
        benchmark_name = benchmark.name.lower().replace("_", "")

        for match in run_only_benchmarks:
            if benchmark_name.startswith(match):
                benchmarks_to_keep.append(benchmark)
                print(f"  keeping benchmark {benchmark.name}, matched {match}")
                break

    if len(benchmarks_to_keep) == 0:
        raise ValueError(
            f"No benchmarks to run after filtering! "
            f"Check the filter criterion '{run_only_benchmarks}' "
            f"and benchmarks {[b.name for b in benchmarks]}."
        )

    return benchmarks_to_keep
