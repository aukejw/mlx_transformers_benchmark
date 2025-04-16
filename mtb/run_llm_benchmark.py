import itertools
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

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

    settings = list(itertools.product(batch_sizes, prompts))
    total_num_iterations = len(settings) * (num_iterations + num_warmup_iterations)

    with tqdm(total=total_num_iterations, position=1, leave=False) as iterator:
        for batch_size, prompt in settings:
            benchmark.set_prompt(prompt=prompt, batch_size=batch_size)
            num_prompt_tokens = benchmark.num_prompt_tokens

            # let us know where we are
            desc = (
                f"  {framework}+{backend}, b={batch_size}, num_prompt_tokens={num_prompt_tokens}, "
                + "warmup {warmup_it} / "
                + f"{num_warmup_iterations}, "
                + "benchmark {it} / "
                + f"{num_iterations}"
            )

            # Run warmup
            iterator.set_description(desc.format(warmup_it=0, it=0))
            start_time = time.perf_counter()
            for warmup_iteration in range(num_warmup_iterations):
                benchmark.run_once()

                iterator.update(1)
                iterator.set_description(
                    desc.format(warmup_it=warmup_iteration + 1, it=0)
                )

            # Run the benchmark
            iterator.set_description(desc.format(warmup_it=num_warmup_iterations, it=0))
            container = Measurements()
            for iteration in range(num_iterations):
                measurement: Dict = benchmark.run_once()
                container.add(measurement)

                iterator.update(1)
                iterator.set_description(
                    desc.format(warmup_it=num_warmup_iterations, it=iteration + 1)
                )

            # Save the measurements
            measurement = dict(
                batch_size=batch_size,
                num_prompt_tokens=num_prompt_tokens,
            )
            for metric in [
                "prompt_tps",
                "prompt_time_sec",
                "num_generated_tokens",
                "generation_tps",
                "current_memory_gb",
            ]:
                measurement[metric] = container.get_mean(metric)

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
    output_path: Union[Path, str],
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

    columns = [
        "name",
        "framework",
        "backend",
        "batch_size",
        "num_prompt_tokens",
        "num_generated_tokens",
        "prompt_tps",  # tokens/sec for processing the prompt
        "prompt_time_sec",  # total time needed to parse prompt, init kv cache
        "generation_tps",  # tokens/sec for generation
        "current_memory_gb",  # memory usage when model and cache are in memory
    ]

    for framework_kwargs in benchmarks_to_run:
        try:
            measurements: List[Dict] = run_benchmark_for_framework(
                **general_kwargs,
                **framework_kwargs,
            )
        except Exception as e:
            print(f"\n  Exception for '{benchmark.name}': {e}")
            continue

        measurements = pd.DataFrame(measurements, columns=columns)
        measurements["name"] = benchmark.name
        for key in framework_kwargs:
            measurements[key] = framework_kwargs[key]

        save_header = not output_path.exists()
        measurements.to_csv(output_path, index=False, mode="a", header=save_header)

    return
