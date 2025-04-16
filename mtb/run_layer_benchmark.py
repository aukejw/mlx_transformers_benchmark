import itertools
import time
from typing import Dict, Iterable, List

import pandas as pd
from tqdm import tqdm

from mtb.layer_benchmarks.base_layer_benchmark import BaseLayerBenchmark


def run_benchmark_for_framework(
    benchmark: BaseLayerBenchmark,
    batch_sizes: Iterable[int],
    sequence_lengths: Iterable[int],
    framework: str,
    backend: str,
    dtype: str,
    num_warmup_iterations: int,
    num_iterations: int,
    min_runtime_ms: int = 500,
    cooldown_time_fraction: float = 0.2,
    compile: bool = False,
) -> List[Dict]:
    """Run a specific benchmark for a specific framework.

    Args:
        benchmark: The benchmark to run.
        framework: The framework to run the benchmark on.
        backend: The backend or compute to use.
        dtype: Identifier for the data type.
        num_warmup_iterations: Number of warmup iterations.
        num_iterations: Number of iterations to run inference for.
        min_runtime_ms: Minimum runtime in milliseconds for the benchmark.
        compile: If true, compile the function before benchmarking.

    Returns:
        A measurement instance, containing the durations of each iteration.

    """
    all_measurements = []

    benchmark.setup(
        framework=framework,
        backend=backend,
        dtype=dtype,
        compile=compile,
    )

    settings = list(itertools.product(batch_sizes, sequence_lengths))
    with tqdm(settings, position=1, leave=False) as iterator:
        for batch_size, sequence_length in iterator:
            benchmark.set_input_tensor(
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
            iterator.set_description(
                f"  {framework}+{backend}, b={batch_size}, seqlen={sequence_length}"
            )

            start_time = time.perf_counter()
            for warmup_iteration in range(num_warmup_iterations):
                benchmark.run_once()

            iteration_time_ms = (
                (time.perf_counter() - start_time) * 1e3 / num_warmup_iterations
            )
            if iteration_time_ms * num_iterations < min_runtime_ms:
                # If iterations are fast, we need to increase the number of iterations
                # for reliability. We set it so the benchmark will take at least some fixed time.
                num_iterations = max(
                    num_iterations, int(min_runtime_ms / iteration_time_ms)
                )

            start_time = time.perf_counter()
            for iteration in range(num_iterations):
                benchmark.run_once()

            duration_ms = (time.perf_counter() - start_time) * 1e3 / num_iterations

            row = dict(
                batch_size=benchmark._batch_size,
                sequence_length=benchmark._sequence_length,
                duration_ms=duration_ms,
            )
            all_measurements.append(row)

            # cooldown
            duration = time.perf_counter() - start_time
            time.sleep(cooldown_time_fraction * duration)

    benchmark.teardown()

    return all_measurements


def run_benchmark(
    benchmark: BaseLayerBenchmark,
    batch_sizes: Iterable[int],
    sequence_lengths: Iterable[int],
    num_warmup_iterations: int = 20,
    num_iterations: int = 50,
    min_runtime_ms: int = 500,
    cooldown_time_fraction: float = 0.2,
    dtype="float32",
    *,
    run_torch_cpu: bool = False,
    run_torch_mps: bool = False,
    run_torch_cuda: bool = False,
    run_mlx_cpu: bool = False,
    run_mlx_metal: bool = False,
    run_mlx_metal_compiled: bool = False,
):
    """Run a benchmark for specific frameworks.

    Args:
        benchmark: The benchmark to run.
        input_shapes: List of input shapes to use.
        num_warmup_iterations: Number of warmup iterations.
        num_iterations: Number of iterations to run inference for.
        min_runtime_ms: Minimum runtime in milliseconds for the benchmark.
        run_torch_cpu: Framework torch, on cpu.
        run_torch_mps: Framework torch, on gpu (mps backend).
        run_torch_cuda: Framework torch, on gpu (cuda backend).
        run_mlx_cpu: Framework mlx, on cpu.
        run_mlx_metal: Framework mlx, on gpu (metal backend).
        run_mlx_metal_compiled: Framework mlx, on gpu (metal backend), compiled.

    Returns:
        pd.DataFrame: A dataframe containing benchmark results.

    """
    general_kwargs = dict(
        benchmark=benchmark,
        batch_sizes=batch_sizes,
        sequence_lengths=sequence_lengths,
        num_warmup_iterations=num_warmup_iterations,
        num_iterations=num_iterations,
        min_runtime_ms=min_runtime_ms,
        cooldown_time_fraction=cooldown_time_fraction,
        dtype=dtype,
    )

    benchmarks_to_run = []
    if run_torch_cpu:
        benchmarks_to_run.append(dict(framework="torch", backend="cpu", compile=False))
    if run_torch_mps:
        benchmarks_to_run.append(dict(framework="torch", backend="mps", compile=False))
    if run_torch_cuda:
        benchmarks_to_run.append(dict(framework="torch", backend="cuda", compile=False))
    if run_mlx_cpu:
        benchmarks_to_run.append(dict(framework="mlx", backend="cpu", compile=False))
    if run_mlx_metal:
        benchmarks_to_run.append(dict(framework="mlx", backend="metal", compile=False))
    if run_mlx_metal_compiled:
        benchmarks_to_run.append(dict(framework="mlx", backend="metal", compile=True))

    benchmark_measurements = []
    for framework_kwargs in benchmarks_to_run:
        measurements: List = run_benchmark_for_framework(
            **general_kwargs,
            **framework_kwargs,
        )

        for row in measurements:
            row.update(
                name=benchmark.name,
                dtype=dtype,
                **framework_kwargs,
            )
            benchmark_measurements.append(row)

    columns = [
        "name",
        "framework",
        "backend",
        "dtype",
        "compile",
        "batch_size",
        "sequence_length",
        "duration_ms",
    ]
    return pd.DataFrame(benchmark_measurements, columns=columns)
