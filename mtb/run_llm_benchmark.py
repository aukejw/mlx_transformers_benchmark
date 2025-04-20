import itertools
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from tqdm import tqdm

from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark
from mtb.measurement import LlmBenchmarkMeasurement, Measurements


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
                f"  {framework}+{backend}, {dtype}, b={batch_size}, num_prompt_tokens={num_prompt_tokens}, "
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
                measurement: LlmBenchmarkMeasurement = benchmark.run_once()
                container.add(measurement.to_dict())

                iterator.update(1)
                iterator.set_description(
                    desc.format(warmup_it=num_warmup_iterations, it=iteration + 1)
                )

            # Save the measurements
            measurement = dict(
                batch_size=batch_size,
                dtype=dtype,
                num_prompt_tokens=num_prompt_tokens,
            )
            for metric_name in container.keys:
                measurement[metric_name] = container.get_mean(metric_name)

            all_measurements.append(measurement)

            # cooldown - don't fry our chip
            total_time = time.perf_counter() - start_time
            time.sleep(cooldown_time_fraction * total_time)

    benchmark.teardown()

    return all_measurements


def run_benchmark(
    benchmark: BaseLLMBenchmark,
    output_path: Union[Path, str],
    batch_sizes: Tuple[int],
    dtypes: Tuple[str],
    prompts: List[str],
    num_warmup_iterations: int = 1,
    num_iterations: int = 5,
    cooldown_time_fraction: float = 0.1,
    *,
    run_torch_cpu: bool = False,
    run_torch_mps: bool = False,
    run_torch_cuda: bool = False,
    run_mlx_cpu: bool = False,
    run_mlx_metal: bool = False,
):
    """Run a benchmark for specific frameworks.

    Each combination of batchsize, prompt, dtype results in one measurement.

    Args:
        benchmark: The benchmark to run.
        output_path: Path to save the benchmark results.
        batch_sizes: List of batch sizes to run.
        dtypes: List of dtypes to run.
        prompts: List of prompts to run.
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
    )

    settings = []
    for dtype in dtypes:
        if run_torch_cpu:
            settings.append(dict(framework="torch", backend="cpu", dtype=dtype))
        if run_torch_mps:
            settings.append(dict(framework="torch", backend="mps", dtype=dtype))
        if run_torch_cuda:
            settings.append(dict(framework="torch", backend="cuda", dtype=dtype))
        if run_mlx_cpu:
            settings.append(dict(framework="mlx", backend="cpu", dtype=dtype))
        if run_mlx_metal:
            settings.append(dict(framework="mlx", backend="metal", dtype=dtype))

    columns = [
        "name",
        "framework",
        "backend",
        "batch_size",
        "dtype",
        "num_prompt_tokens",
        "num_generated_tokens",
        "prompt_tps",  # tokens/sec for processing the prompt
        "prompt_time_sec",  # total time needed to parse prompt, init kv cache
        "generation_tps",  # tokens/sec for generation
        "generation_time_sec",  # total time needed for generation, excl. prompting
    ]

    for framework_kwargs in settings:
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
