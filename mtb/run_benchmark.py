import time

import pandas as pd

from mtb.benchmarks.base_benchmark import BaseBenchmark
from mtb.measurement import Measurement


def run_benchmark_for_framework(
    benchmark: BaseBenchmark,
    framework: str,
    backend: str,
    dtype: str,
    num_warmup_iterations: int,
    num_iterations: int,
    num_repeats: int,
    compile: bool = False,
) -> Measurement:
    """Run a specific benchmark for a specific framework.

    Args:
        benchmark (BaseBenchmark): The benchmark to run.
        framework (str): The framework to run the benchmark on.
        backend (str): The backend or compute to use.
        dtype (str): Identifier for the data type.
        num_warmup_iterations (int): Number of warmup iterations.
        num_iterations (int): Number of iterations to run inference for.
        repeats (int): Number of times to repeat the benchmark.
        compile (bool): If true, compile the function before benchmarking.

    Returns:
        A measurement instance, containing the durations of each iteration.

    """
    benchmark.setup(
        framework=framework,
        backend=backend,
        dtype=dtype,
        compile=compile,
    )

    for warmup_iteration in range(num_warmup_iterations):
        benchmark.run_once(
            framework=framework,
            backend=backend,
        )

    measurements = []
    for repeat_index in range(num_repeats):
        start_time = time.perf_counter()
        for iteration in range(num_iterations):
            benchmark.run_once(
                framework=framework,
                backend=backend,
            )
        duration_ms = (time.perf_counter() - start_time) * 1e3 / num_iterations
        measurements.append(duration_ms)

    benchmark.teardown(
        framework=framework,
        backend=backend,
    )

    return Measurement(measurements=measurements)


def run_benchmark(
    benchmark: BaseBenchmark,
    num_warmup_iterations: int = 20,
    num_iterations: int = 50,
    num_repeats: int = 1,
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
        num_warmup_iterations: Number of warmup iterations.
        num_iterations: Number of iterations to run inference for.
        num_repeats: Number of times to repeat the timing.
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
        num_warmup_iterations=num_warmup_iterations,
        num_iterations=num_iterations,
        num_repeats=num_repeats,
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
        measurement: Measurement = run_benchmark_for_framework(
            **general_kwargs,
            **framework_kwargs,
        )
        row = dict(
            name=benchmark.name,
            batch_size=benchmark.input_shape[0],
            sequence_length=benchmark.input_shape[1],
            **framework_kwargs,
            median_ms=measurement.median,
            mean_ms=measurement.mean,
            std_ms=measurement.std,
        )
        benchmark_measurements.append(row)

    columns = [
        "name",
        "framework",
        "backend",
        "compile",
        "batch_size",
        "sequence_length",
        "num_warmup_iterations",
        "num_iterations",
        "num_repeats",
        "median_ms",
        "mean_ms",
        "std_ms",
    ]
    return pd.DataFrame(benchmark_measurements, columns=columns)
