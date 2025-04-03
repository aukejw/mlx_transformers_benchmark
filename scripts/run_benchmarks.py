import itertools
import time
from pathlib import Path
from typing import Tuple, Union

import fire
import pandas as pd
from tqdm import tqdm

import mtb as mtb
import mtb.benchmarks as mtb_bench
from mtb.file_io import create_benchmark_output_dir
from mtb.run_benchmark import run_benchmark

DEFAULT_OUTPUT_ROOT = mtb.REPO_ROOT / "measurements"


def main(
    output_root: Union[str, Path] = DEFAULT_OUTPUT_ROOT,
    num_warmup_iterations: int = 20,
    num_iterations: int = 50,
    dtype: str = "float32",
    cooldown_time_fraction: float = 0.10,
    batch_sizes: Tuple = (64, 32, 16, 8, 1),
    sequence_lengths: Tuple = (512, 256, 128, 64),
    feature_dim: int = 768,
    *,
    run_torch_mps: bool = True,
    run_mlx_metal: bool = True,
    run_mlx_metal_compiled: bool = True,
    run_torch_cpu: bool = False,
    run_torch_cuda: bool = False,
    run_mlx_cpu: bool = False,
):
    """Run benchmarks. By default, we always run torch mps on GPU, and mlx on metal.

    To avoid frying the hardware, we add a small cooldown during which the chips
    should (mostly) idle. A cooldown of 10% of the duration of the task results
    in a 95°C peak GPU temperature on a Macbook M4 Pro, but YMMV.

    By default, we run torch with MPS becakend, and MLX in compiled and
    non-compiled mode.

    """
    output_dir = create_benchmark_output_dir(
        output_root=output_root,
        benchmark_settings=dict(
            num_warmup_iterations=num_warmup_iterations,
            num_iterations=num_iterations,
            dtype=dtype,
        ),
    )
    print(f"Output directory: '{output_dir}'")

    benchmarks = []
    for batch_size, sequence_length in itertools.product(batch_sizes, sequence_lengths):
        input_shape = (batch_size, sequence_length, feature_dim)

        kwargs = dict(
            input_shape=input_shape,
        )

        benchmarks.extend(
            [
                mtb_bench.LayerNormBenchmark(**kwargs),
                mtb_bench.LinearBenchmark(**kwargs),
                mtb_bench.MhsaBenchmark(num_heads=1, **kwargs),
                mtb_bench.MhsaBenchmark(num_heads=8, **kwargs),
                mtb_bench.TransformerEncoderLayerBenchmark(
                    mask_type=None,
                    num_heads=8,
                    **kwargs,
                ),
                mtb_bench.TransformerEncoderLayerBenchmark(
                    mask_type="causal",
                    num_heads=8,
                    **kwargs,
                ),
                mtb_bench.TransformerDecoderLayerBenchmark(
                    mask_type=None,
                    num_heads=8,
                    **kwargs,
                ),
                mtb_bench.TransformerDecoderLayerBenchmark(
                    mask_type="causal",
                    num_heads=8,
                    **kwargs,
                ),
            ]
        )

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
        duration_s = time.perf_counter() - start_time

        # Save measurements after each benchmark to avoid losing data on interruption
        output_path = output_dir / "benchmark_results.csv"
        save_header = not output_path.exists()
        results.to_csv(output_path, index=False, mode="a", header=save_header)

        # Cooldown is a fraction of the duration
        print(f"Duration {duration_s}")
        time.sleep(cooldown_time_fraction * duration_s)

    print(f"Saved measurements to '{output_path}'")
    return


if __name__ == "__main__":
    fire.Fire(main)
