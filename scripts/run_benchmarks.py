import itertools
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import fire
import pandas as pd
from tqdm import tqdm

import mtb as mtb
import mtb.benchmarks as mtb_bench
from mtb.file_io import create_benchmark_output_dir
from mtb.run_benchmark import run_benchmark
from mtb.select_benchmarks import filter_benchmarks

DEFAULT_OUTPUT_ROOT = mtb.REPO_ROOT / "measurements"


def main(
    output_root: Union[str, Path] = DEFAULT_OUTPUT_ROOT,
    num_warmup_iterations: int = 20,
    num_iterations: int = 50,
    num_repeats: int = 1,
    dtype: str = "float32",
    cooldown_time_fraction: float = 0.10,
    batch_sizes: Tuple = (64, 32, 16, 8, 1),
    sequence_lengths: Tuple = (512, 256, 128, 64),
    num_attention_heads: Tuple = (1, 8),
    mask_types: Tuple = (None, "causal"),
    feature_dim: int = 768,
    *,
    run_torch_mps: bool = True,
    run_mlx_metal: bool = True,
    run_mlx_metal_compiled: bool = True,
    run_torch_cpu: bool = False,
    run_torch_cuda: bool = False,
    run_mlx_cpu: bool = False,
    run_only_benchmarks: Optional[List[str]] = None,
):
    """Run benchmarks. By default, we always run torch mps on GPU, and mlx on metal.

    To avoid frying the hardware, we add a small cooldown during which the chips
    should (mostly) idle. A cooldown of 10% of the duration of the task results
    in a 95°C peak GPU temperature on a Macbook M4 Pro, but YMMV.

    By default, we run torch with MPS becakend, and MLX in compiled and
    non-compiled mode.

    """
    # Set up benchmarks
    benchmarks = []
    for batch_size, sequence_length in itertools.product(batch_sizes, sequence_lengths):
        input_shape = (batch_size, sequence_length, feature_dim)

        # add regular benchmarks
        kwargs = dict(
            input_shape=input_shape,
        )
        regular_benchmarks = [
            mtb_bench.LayerNormBenchmark(**kwargs),
            mtb_bench.LinearBenchmark(**kwargs),
            mtb_bench.SoftmaxBenchmark(**kwargs),
        ]
        benchmarks.extend(regular_benchmarks)

        # add benchmarks with attention heads + masks
        for num_heads, mask_type in itertools.product(num_attention_heads, mask_types):
            kwargs["num_heads"] = num_heads
            kwargs["mask_type"] = mask_type

            attention_benchmarks = [
                mtb_bench.MhsaBenchmark(**kwargs),
                mtb_bench.MhsaBenchmark(**kwargs),
                mtb_bench.TransformerEncoderLayerBenchmark(**kwargs),
                mtb_bench.TransformerEncoderLayerBenchmark(**kwargs),
                mtb_bench.TransformerDecoderLayerBenchmark(**kwargs),
                mtb_bench.TransformerDecoderLayerBenchmark(**kwargs),
            ]
            benchmarks.extend(attention_benchmarks)

    # Filter benchmarks if specified
    if run_only_benchmarks is not None:
        num_benchmarks = len(benchmarks)
        benchmarks = filter_benchmarks(
            benchmarks=benchmarks,
            run_only_benchmarks=run_only_benchmarks,
        )
        if len(benchmarks) == 0:
            raise ValueError(
                f"No benchmarks to run! Check the filter: {run_only_benchmarks}."
            )

        print(f"Running {len(benchmarks)} out of {num_benchmarks} benchmarks")

    # Create output directory for measurements
    output_dir = create_benchmark_output_dir(
        output_root=output_root,
        benchmark_settings=dict(
            num_warmup_iterations=num_warmup_iterations,
            num_iterations=num_iterations,
            num_repeats=num_repeats,
            dtype=dtype,
            run_only_benchmarks=run_only_benchmarks,
        ),
    )
    print(f"Output directory: '{output_dir}'")

    # Run
    iterator = tqdm(benchmarks)

    all_results = []
    for benchmark in iterator:
        iterator.set_description(
            f"Timing {benchmark.name}, input_shape={benchmark.input_shape}"
        )
        start_time = time.perf_counter()

        try:
            results: pd.DataFrame = run_benchmark(
                benchmark=benchmark,
                num_warmup_iterations=num_warmup_iterations,
                num_iterations=num_iterations,
                num_repeats=num_repeats,
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
        duration_seconds = time.perf_counter() - start_time

        # Save measurements after each benchmark to avoid losing data on interruption
        output_path = output_dir / "benchmark_results.csv"
        save_header = not output_path.exists()
        results.to_csv(output_path, index=False, mode="a", header=save_header)

        # Cooldown is a fraction of the task duration -- let's not fry your chips
        time.sleep(cooldown_time_fraction * duration_seconds)

    print(f"Saved measurements to '{output_path}'")
    return


if __name__ == "__main__":
    fire.Fire(main)
