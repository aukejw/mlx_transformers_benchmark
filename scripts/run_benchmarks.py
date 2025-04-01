import itertools
from pathlib import Path
from typing import Union

import fire
import pandas as pd
from tqdm import tqdm

import mlx_transformers_benchmark as mtb
import mlx_transformers_benchmark.benchmarks as mtb_bench
from mlx_transformers_benchmark.file_io import create_benchmark_output_dir
from mlx_transformers_benchmark.run_benchmark import run_benchmark

DEFAULT_OUTPUT_ROOT = mtb.REPO_ROOT / "measurements"


def main(
    output_root: Union[str, Path] = DEFAULT_OUTPUT_ROOT,
    num_warmup_iterations: int = 30,
    num_iterations: int = 100,
    dtype: str = "float32",
    run_torch_mps: bool = True,
    run_mlx_metal: bool = True,
    *,
    run_torch_cpu: bool = False,
    run_torch_cuda: bool = False,
    run_mlx_cpu: bool = False,
    run_mlx_metal_compiled: bool = False,
):
    """Run benchmarks. By default, we always run torch mps on GPU, and mlx on metal."""

    output_dir = create_benchmark_output_dir(
        output_root=output_root,
        benchmark_settings=dict(
            num_warmup_iterations=num_warmup_iterations,
            num_iterations=num_iterations,
            dtype=dtype,
        ),
    )
    print(f"Output directory: '{output_dir}'")

    batch_sizes = [128, 64, 32, 16, 1]
    sequence_lengths = [512, 256, 128, 64]

    benchmarks = []
    for batch_size, sequence_length in itertools.product(batch_sizes, sequence_lengths):
        input_shape = (batch_size, sequence_length, 1024)

        benchmarks.extend(
            [
                mtb_bench.MhsaBenchmark(input_shapes=[input_shape]),
                mtb_bench.LayerNormBenchmark(input_shapes=[input_shape]),
                mtb_bench.LinearBenchmark(input_shapes=[input_shape]),
                mtb_bench.TransformerEncoderLayerBenchmark(input_shapes=[input_shape]),
            ]
        )

    iterator = tqdm(benchmarks)

    all_results = []
    for benchmark in iterator:
        iterator.set_description(
            f"Timing {benchmark.name}, input_shapes={benchmark.input_shapes}"
        )
        try:
            results: pd.DataFrame = run_benchmark(
                benchmark=benchmark,
                num_warmup_iterations=num_warmup_iterations,
                num_iterations=num_iterations,
                dtype=dtype,
                run_torch_cpu=run_torch_cpu,
                run_torch_mps=run_torch_mps,
                run_torch_cuda=run_torch_cuda,
                run_mlx_cpu=run_mlx_cpu,
                run_mlx_metal=run_mlx_metal,
                run_mlx_metal_compiled=run_mlx_metal_compiled,
            )
        except Exception as e:
            print(f"Error running benchmark '{benchmark}': {e}")
            continue

        all_results.append(results)

        # Save measurements after each benchmark to avoid losing data on interruption
        output_path = output_dir / "benchmark_results.csv"
        save_header = not output_path.exists()
        results.to_csv(output_path, index=False, mode="a", header=save_header)

    print(f"Saved measurements to '{output_path}'")
    return


if __name__ == "__main__":
    fire.Fire(main)
