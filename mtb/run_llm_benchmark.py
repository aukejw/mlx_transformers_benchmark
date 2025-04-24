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
from mtb.llm_benchmarks.torch_llm_benchmark import TorchLlmBenchmark
from mtb.measurement import LlmBenchmarkMeasurement, Measurements
from mtb.memory import estimate_model_size, get_available_ram_gb


def run_benchmark(
    model_spec: ModelSpec,
    output_path: Union[Path, str],
    batch_sizes: Tuple[int],
    dtypes: Tuple[str],
    prompts: List[str],
    num_warmup_iterations: int = 1,
    num_iterations: int = 5,
    max_num_tokens: int = 100,
    cooldown_time_fraction: float = 0.1,
    *,
    run_torch_cpu: bool = False,
    run_torch_mps: bool = False,
    run_torch_cuda: bool = False,
    run_mlx_cpu: bool = False,
    run_mlx_metal: bool = False,
    run_lmstudio_metal: bool = False,
):
    """Run a benchmark for specific models.

    Each combination of batchsize, prompt, dtype results in one measurement.

    Args:
        model_spec: The specs for the model to benchmark.
        output_path: Path to save benchmark results.
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
        run_lmstudio_metal: Framework lmstudio, on gpu (llama.cpp metal backend).

    Returns:
        pd.DataFrame: A dataframe containing benchmark results.

    """
    available_memory = get_available_ram_gb()

    settings = []
    for dtype in dtypes:
        # Check if we can run it
        memory_needed_gb = estimate_model_size(
            num_params=model_spec.num_params,
            dtype=dtype,
        )
        if memory_needed_gb > available_memory:
            print(
                f"Skipping model '{model_spec.name}' for dtype {dtype}: "
                f"it needs {memory_needed_gb:.3f} GB memory just to load the model, "
                f"but only {available_memory:.3f} GB is available."
            )
            continue

        # If so, define the available benchmark settings
        setting = dict(dtype=dtype)
        if model_spec.has_model_id(framework="torch", dtype=dtype):
            if run_torch_cpu:
                setting.update(framework="torch", backend="cpu")
            if run_torch_mps:
                setting.update(framework="torch", backend="mps")
            if run_torch_cuda:
                setting.update(framework="torch", backend="cuda")

        if model_spec.has_model_id("mlx", dtype):
            if run_mlx_cpu and dtype in model_spec.model_ids["mlx"]:
                setting.update(framework="mlx", backend="cpu")
            if run_mlx_metal and dtype in model_spec.model_ids["mlx"]:
                setting.update(framework="mlx", backend="metal")

        if model_spec.has_model_id("lmstudio", dtype):
            if run_lmstudio_metal and dtype in model_spec.model_ids["lmstudio"]:
                setting.update(framework="lmstudio", backend="metal+llama.cpp")

        if "framework" in setting:
            settings.append(setting)

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
    ]

    for framework_kwargs in settings:
        benchmark: BaseLLMBenchmark = create_benchmark(
            model_spec=model_spec,
            max_num_tokens=max_num_tokens,
            **framework_kwargs,
        )

        try:
            measurements: List[Dict] = run_benchmark_for_framework(
                benchmark=benchmark,
                batch_sizes=batch_sizes,
                prompts=prompts,
                num_warmup_iterations=num_warmup_iterations,
                num_iterations=num_iterations,
                cooldown_time_fraction=cooldown_time_fraction,
            )
        except Exception as e:
            print(f"\n  Exception for '{benchmark.name}': {e}")
            continue

        # Save measurements to csv
        measurements = pd.DataFrame(measurements, columns=csv_columns)
        measurements.to_csv(
            output_path,
            index=False,
            mode="a",
            header=(not output_path.exists()),
        )

    return


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
    prompts: List[str],
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

    settings = list(itertools.product(batch_sizes, prompts))
    total_num_iterations = len(settings) * (num_iterations + num_warmup_iterations)

    all_measurements = []
    with tqdm(total=total_num_iterations, position=1, leave=False) as iterator:
        for setting_index, (batch_size, prompt) in enumerate(settings):
            assert batch_size == 1, "Batch size > 1 not supported yet."

            prompt_tokens = benchmark.format_prompt(prompt=prompt)
            num_prompt_tokens = len(prompt_tokens)

            # let us know where we are
            desc = (
                f"  {benchmark.framework}+{benchmark.backend}, {benchmark.dtype}, "
                + f"setting={setting_index}/{len(settings)}, "
                + "warmup {warmup_it} / "
                + f"{num_warmup_iterations}, "
                + "benchmark {it} / "
                + f"{num_iterations}"
            )

            # Run warmup
            iterator.set_description(desc.format(warmup_it=0, it=0))
            start_time = time.perf_counter()
            for warmup_iteration in range(num_warmup_iterations):
                benchmark.run_once(prompt=prompt)

                iterator.update(1)
                iterator.set_description(
                    desc.format(warmup_it=warmup_iteration + 1, it=0)
                )

            # Run the benchmark
            iterator.set_description(desc.format(warmup_it=num_warmup_iterations, it=0))
            container = Measurements()
            for iteration in range(num_iterations):
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
                num_prompt_tokens=num_prompt_tokens,
            )
            for metric_name in container.keys:
                measurement[metric_name] = container.get_mean(metric_name)

            all_measurements.append(measurement)

            # Cooldown - don't fry our chip
            total_time = time.perf_counter() - start_time
            time.sleep(cooldown_time_fraction * total_time)

    benchmark.teardown()

    return all_measurements
