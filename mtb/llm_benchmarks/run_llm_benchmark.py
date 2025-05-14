import itertools
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from tqdm import tqdm

from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark
from mtb.llm_benchmarks.lmstudio_llm_benchmark import LMStudioLlmBenchmark
from mtb.llm_benchmarks.mlx_llm_benchmark import MlxLlmBenchmark
from mtb.llm_benchmarks.models.base import ModelSpec
from mtb.llm_benchmarks.ollama_llm_benchmark import OllamaLlmBenchmark
from mtb.llm_benchmarks.torch_llm_benchmark import TorchLlmBenchmark
from mtb.measurement import LlmBenchmarkMeasurement, Measurements
from mtb.prompts import find_prompt_for_llm_benchmark


def run_benchmark(
    model_spec: Dict,
    framework: str,
    backend: str,
    dtype: str,
    output_path: Union[Path, str],
    batch_sizes: Tuple[int],
    prompt_lengths: List[str],
    num_warmup_iterations: int = 1,
    num_iterations: int = 5,
    max_num_tokens: int = 100,
    cooldown_time_fraction: float = 0.1,
):
    """Run a benchmark for a specific model (and, by definition, dtype).

    Each combination of batchsize, prompt results in one measurement.

    Args:
        benchmark_config: Configuration for the benchmark, including model_id.
        output_path: Path to save benchmark results.
        batch_sizes: List of batch sizes to run.
        prompt_lengths: List of lengths of prompts to use.
        num_warmup_iterations: Number of warmup iterations.
        num_iterations: Number of iterations to run generation for.
        cooldown_time_fraction: Fraction of time to wait after each benchmark.
            This is to avoid overheating the GPU.
        max_num_tokens: Maximum number of tokens to generate.

    Returns:
        pd.DataFrame: A dataframe containing benchmark results.

    """
    csv_columns = [
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
        "peak_memory_gib",  # peak memory usage in GiB
    ]

    benchmark: BaseLLMBenchmark = create_benchmark(
        model_spec=model_spec,
        framework=framework,
        backend=backend,
        dtype=dtype,
        max_num_tokens=max_num_tokens,
    )

    try:
        measurements: List[Dict] = run_benchmark_for_framework(
            benchmark=benchmark,
            batch_sizes=batch_sizes,
            prompt_lengths=prompt_lengths,
            num_warmup_iterations=num_warmup_iterations,
            num_iterations=num_iterations,
            cooldown_time_fraction=cooldown_time_fraction,
        )

        # Save measurements to csv
        measurements = pd.DataFrame(measurements, columns=csv_columns)
        measurements.to_csv(
            output_path,
            index=False,
            mode="a",
            header=(not output_path.exists()),
        )

    except Exception as e:
        print(f"\n  Exception for '{benchmark.name}': {e}")

    return pd.read_csv(output_path)


def create_benchmark(
    model_spec: ModelSpec,
    framework: str,
    backend: str,
    dtype: str,
    max_num_tokens: int = 100,
) -> BaseLLMBenchmark:
    """Create a benchmark for a specific task."""

    if framework == "torch":
        benchmark_class = TorchLlmBenchmark
    elif framework == "mlx":
        benchmark_class = MlxLlmBenchmark
    elif framework == "lmstudio":
        benchmark_class = LMStudioLlmBenchmark
    elif framework == "ollama":
        benchmark_class = OllamaLlmBenchmark
    else:
        raise NotImplementedError(f"Framework not supported: {framework}. ")

    model_id = model_spec.model_ids[framework][dtype]

    benchmark = benchmark_class(
        name=model_spec.name,
        prompt_formatter=model_spec.prompt_formatter,
        model_id=model_id,
        backend=backend,
        dtype=dtype,
        max_num_tokens=max_num_tokens,
    )
    return benchmark


def run_benchmark_for_framework(
    benchmark: BaseLLMBenchmark,
    batch_sizes: Tuple[int],
    prompt_lengths: List[int],
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
    benchmark.setup()

    settings = list(itertools.product(batch_sizes, prompt_lengths))
    total_num_iterations = len(settings) * (num_iterations + num_warmup_iterations)

    all_measurements = []
    with tqdm(total=total_num_iterations, position=1, leave=False) as iterator:
        for batch_size, num_prompt_tokens in settings:
            assert batch_size == 1, "Batch size > 1 not supported yet."

            # Let us know where we are
            desc = (
                f"  {benchmark.framework}+{benchmark.backend}, {benchmark.dtype}, "
                + f"B={batch_size}, L={num_prompt_tokens}, "
                + "warmup {warmup_it} / "
                + f"{num_warmup_iterations}, "
                + "benchmark {it} / "
                + f"{num_iterations}"
            )

            # Run warmup
            iterator.set_description(desc.format(warmup_it=0, it=0))
            start_time = time.perf_counter()
            for warmup_iteration in range(num_warmup_iterations):
                prompt = find_prompt_for_llm_benchmark(
                    benchmark=benchmark,
                    num_tokens=num_prompt_tokens,
                )
                benchmark.run_once(prompt=prompt)

                iterator.update(1)
                iterator.set_description(
                    desc.format(warmup_it=warmup_iteration + 1, it=0)
                )

            # Run the benchmark
            iterator.set_description(desc.format(warmup_it=num_warmup_iterations, it=0))
            container = Measurements()
            for iteration in range(num_iterations):
                prompt = find_prompt_for_llm_benchmark(
                    benchmark=benchmark,
                    num_tokens=num_prompt_tokens,
                )
                measurement: LlmBenchmarkMeasurement = benchmark.run_once(prompt=prompt)
                container.add(measurement.to_dict())

                iterator.update(1)
                iterator.set_description(
                    desc.format(warmup_it=num_warmup_iterations, it=iteration + 1)
                )

            # Save the (averaged) measurements
            measurement = dict(
                name=benchmark.name,
                framework=benchmark.framework,
                backend=benchmark.backend,
                dtype=benchmark.dtype,
                batch_size=batch_size,
                num_prompt_tokens=int(num_prompt_tokens),
            )
            for metric_name in container.keys:
                if metric_name not in measurement:
                    measurement[metric_name] = container.get_mean(metric_name)

            all_measurements.append(measurement)

            # Cooldown - don't fry our chip
            total_time = time.perf_counter() - start_time
            time.sleep(cooldown_time_fraction * total_time)

    benchmark.teardown()

    return all_measurements
