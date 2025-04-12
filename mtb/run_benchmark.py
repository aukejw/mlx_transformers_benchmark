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
    min_runtime_ms: int = 500,
    compile: bool = False,
) -> Measurement:
    """Run a specific benchmark for a specific framework.

    Args:
        benchmark: The benchmark to run.
        framework: The framework to run the benchmark on.
        backend: The backend or compute to use.
        dtype: Identifier for the data type.
        num_warmup_iterations: Number of warmup iterations.
        num_iterations: Number of iterations to run inference for.
        repeats: Number of times to repeat the benchmark.
        min_runtime_ms: Minimum runtime in milliseconds for the benchmark.
        compile: If true, compile the function before benchmarking.

    Returns:
        A measurement instance, containing the durations of each iteration.

    """
    benchmark.setup(
        framework=framework,
        backend=backend,
        dtype=dtype,
        compile=compile,
    )

    start_time = time.perf_counter()
    for warmup_iteration in range(num_warmup_iterations):
        benchmark.run_once()

    iteration_time_ms = (time.perf_counter() - start_time) * 1e3 / num_warmup_iterations
    if iteration_time_ms * num_iterations < min_runtime_ms:
        # If iterations are fast, we need to increase the number of iterations
        # for reliability. We set it so the benchmark will take at least some fixed time.
        num_iterations = max(num_iterations, int(min_runtime_ms / iteration_time_ms))

    measurements = []
    for repeat_index in range(num_repeats):
        start_time = time.perf_counter()

        for iteration in range(num_iterations):
            benchmark.run_once()

        duration_ms = (time.perf_counter() - start_time) * 1e3 / num_iterations
        measurements.append(duration_ms)

    benchmark.teardown()

    return Measurement(measurements=measurements)


def run_benchmark(
    benchmark: BaseBenchmark,
    num_warmup_iterations: int = 20,
    num_iterations: int = 50,
    num_repeats: int = 1,
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
        num_warmup_iterations: Number of warmup iterations.
        num_iterations: Number of iterations to run inference for.
        num_repeats: Number of times to repeat the timing.
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
        num_warmup_iterations=num_warmup_iterations,
        num_iterations=num_iterations,
        num_repeats=num_repeats,
        min_runtime_ms=min_runtime_ms,
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
        start_time = time.perf_counter()

        measurement: Measurement = run_benchmark_for_framework(
            **general_kwargs,
            **framework_kwargs,
        )
        duration_seconds = time.perf_counter() - start_time

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

        # Cooldown is a fraction of the task duration -- let's not fry your chips
        time.sleep(cooldown_time_fraction * duration_seconds)

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
