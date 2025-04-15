import itertools
import time
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark
from mtb.measurement import Measurements


def run_benchmark_for_framework(
    benchmark: BaseLLMBenchmark,
    batch_sizes: Tuple[int],
    prompts: List[str],
    framework: str,
    backend: str,
    dtype: str,
    num_warmup_iterations: int,
    num_iterations: int,
    cooldown_time_fraction: float,
) -> Dict[str, float]:
    """Run a specific benchmark for a specific framework.

    Args:
        benchmark: The benchmark to run.
        framework: The framework to run the benchmark on.
        backend: The backend or compute to use.
        dtype: Identifier for the data type.
        num_warmup_iterations: Number of warmup iterations.
        num_iterations: Number of iterations to run generation for.

    Returns:
        List of measurements containing benchmark results.

    """
    all_measurements = []

    benchmark.setup(
        framework=framework,
        backend=backend,
        dtype=dtype,
    )

    for batch_size, prompt in itertools.product(batch_sizes, prompts):
        benchmark.set_prompt(prompt=prompt, batch_size=batch_size)
        num_prompt_tokens = benchmark.num_prompt_tokens
        measurements_container = Measurements()

        start_time = time.perf_counter()

        # let us know where we are
        with tqdm(total=num_iterations + num_warmup_iterations) as iterator:
            desc = (
                f"  b={batch_size}, num_prompt_tokens={num_prompt_tokens}, "
                + "warmup {warmup_it} / "
                + f"{num_warmup_iterations}, "
                + "benchmark {it} / "
                + f"{num_iterations}"
            )

            iterator.set_description(desc.format(warmup_it=0, it=0))
            for warmup_iteration in range(num_warmup_iterations):
                benchmark.run_once()

                iterator.update(1)
                iterator.set_description(
                    desc.format(warmup_it=warmup_iteration + 1, it=0)
                )

            iterator.set_description(desc.format(warmup_it=num_warmup_iterations, it=0))
            for iteration in range(num_iterations):
                measurement: Dict = benchmark.run_once()
                measurements_container.add(measurement)

                iterator.update(1)
                iterator.set_description(
                    desc.format(warmup_it=num_warmup_iterations, it=iteration + 1)
                )

        # Save the measurements
        measurement = dict(
            batch_size=batch_size,
            num_prompt_tokens=num_prompt_tokens,
            generation_tps=measurements_container.get_mean("generation_tps"),
            prompt_tps=measurements_container.get_mean("prompt_tps"),
            peak_memory_gb=measurements_container.get_mean("peak_memory_gb"),
        )
        all_measurements.append(measurement)

        # cooldown - don't fry our chip
        total_time = time.perf_counter() - start_time
        time.sleep(cooldown_time_fraction * total_time)

    benchmark.teardown()

    return all_measurements


def run_benchmark(
    benchmark: BaseLLMBenchmark,
    batch_sizes: Tuple[int],
    prompts: List[str],
    num_warmup_iterations: int = 1,
    num_iterations: int = 10,
    cooldown_time_fraction: float = 0.2,
    dtype="float32",
    *,
    run_torch_cpu: bool = False,
    run_torch_mps: bool = False,
    run_torch_cuda: bool = False,
    run_mlx_cpu: bool = False,
    run_mlx_metal: bool = False,
):
    """Run a benchmark for specific frameworks.

    Args:
        benchmark: The benchmark to run.
        prompts: List of prompts to run. Each prompt results in one measurement.
        num_warmup_iterations: Number of warmup iterations.
        num_iterations: Number of iterations to run generation for.
        run_torch_cpu: Framework torch, on cpu.
        run_torch_mps: Framework torch, on gpu (mps backend).
        run_torch_cuda: Framework torch, on gpu (cuda backend).
        run_mlx_cpu: Framework mlx, on cpu.
        run_mlx_metal: Framework mlx, on gpu (metal backend).

    Returns:
        pd.DataFrame: A dataframe containing benchmark results.

    """
    general_kwargs = dict(
        benchmark=benchmark,
        batch_sizes=batch_sizes,
        prompts=prompts,
        num_warmup_iterations=num_warmup_iterations,
        num_iterations=num_iterations,
        cooldown_time_fraction=cooldown_time_fraction,
        dtype=dtype,
    )

    benchmarks_to_run = []
    if run_torch_cpu:
        benchmarks_to_run.append(dict(framework="torch", backend="cpu"))
    if run_torch_mps:
        benchmarks_to_run.append(dict(framework="torch", backend="mps"))
    if run_torch_cuda:
        benchmarks_to_run.append(dict(framework="torch", backend="cuda"))
    if run_mlx_cpu:
        benchmarks_to_run.append(dict(framework="mlx", backend="cpu"))
    if run_mlx_metal:
        benchmarks_to_run.append(dict(framework="mlx", backend="metal"))

    benchmark_measurements = []
    for framework_kwargs in benchmarks_to_run:
        measurements: List[Dict] = run_benchmark_for_framework(
            **general_kwargs,
            **framework_kwargs,
        )
        for row in measurements:
            row.update(
                name=benchmark.name,
                **framework_kwargs,
            )
            benchmark_measurements.append(row)

    columns = [
        "name",
        "framework",
        "backend",
        "batch_size",
        "num_prompt_tokens",
        "num_warmup_iterations",
        "num_iterations",
        "prompt_tps",  # tokens/sec for processing the prompt
        "generation_tps",  # tokens/sec for generation
        "peak_memory_gb",  # peak memory usage
    ]
    return pd.DataFrame(benchmark_measurements, columns=columns)
