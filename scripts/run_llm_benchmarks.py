from pathlib import Path
from typing import List, Optional, Tuple, Union

import fire
from tqdm import tqdm

import mtb as mtb
import mtb.llm_benchmarks as mtb_bench
from mtb.file_io import create_benchmark_output_dir
from mtb.run_llm_benchmark import run_benchmark
from mtb.select_benchmarks import filter_benchmarks

DEFAULT_OUTPUT_ROOT = mtb.REPO_ROOT / "measurements" / "llm_benchmarks"


def main(
    output_root: Union[str, Path] = DEFAULT_OUTPUT_ROOT,
    num_warmup_iterations: int = 1,
    num_iterations: int = 5,
    batch_sizes: Tuple = (1,),
    dtypes: str = ("bfloat16", "int8", "int4"),
    max_num_tokens: int = 100,
    enable_hf_progressbar: bool = False,
    cooldown_time_fraction: float = 0.05,
    *,
    run_only_benchmarks: Optional[List[str]] = None,
    run_mlx_metal: bool = True,
    run_torch_mps: bool = False,
    run_torch_cpu: bool = False,
    run_torch_cuda: bool = False,
    run_mlx_cpu: bool = False,
):
    """Run LLM benchmarks.

    To avoid frying the hardware, we add a small cooldown during which the chips
    should (mostly) idle. A cooldown of 10% of the duration of the task results
    in a 95°C peak GPU temperature on a Macbook M4 Pro, but YMMV.

    By default, we run MLX with Metal backend only.

    """
    from mtb.hf_utils import set_hf_home

    set_hf_home(enable_hf_progressbar=enable_hf_progressbar)

    prompts = [
        "Repeat the following sequence: " + ", ".join(str(i) for i in range(1_000)),
        "Repeat the following sequence: " + ", ".join(str(i) for i in range(500)),
        "Repeat the following sequence: " + ", ".join(str(i) for i in range(100)),
        "Write a story about Einstein",
    ]

    # Set up benchmarks
    kwargs = dict(
        max_num_tokens=max_num_tokens,
    )
    benchmarks = [
        # gemma 3
        mtb_bench.Gemma3_1B_it_Benchmark(**kwargs),
        mtb_bench.Gemma3_4B_it_Benchmark(**kwargs),
        # gemma 3 qat
        mtb_bench.Gemma3_1B_it_QAT_Benchmark(**kwargs),
        mtb_bench.Gemma3_4B_it_QAT_Benchmark(**kwargs),
        mtb_bench.Gemma3_12B_it_QAT_Benchmark(**kwargs),
        # qwen 2.5
        mtb_bench.Qwen2p5_0p5B_it_Benchmark(**kwargs),
        mtb_bench.Qwen2p5_3B_it_Benchmark(**kwargs),
        # qwen 2.5 coder
        mtb_bench.Qwen2p5_Coder_0p5B_it_Benchmark(**kwargs),
        mtb_bench.Qwen2p5_Coder_3B_it_Benchmark(**kwargs),
    ]

    # Filter benchmarks if specified
    if run_only_benchmarks is not None:
        num_benchmarks = len(benchmarks)
        benchmarks = filter_benchmarks(
            benchmarks=benchmarks,
            run_only_benchmarks=run_only_benchmarks,
        )
        print(f"Running {len(benchmarks)} out of {num_benchmarks} benchmarks")

    # Create output directory for measurements
    output_dir = create_benchmark_output_dir(
        output_root=output_root,
        benchmark_settings=dict(
            num_warmup_iterations=num_warmup_iterations,
            num_iterations=num_iterations,
            dtypes=dtypes,
            run_only_benchmarks=run_only_benchmarks,
        ),
    )
    output_path = output_dir / "benchmark_results.csv"
    print(f"Output directory: '{output_dir}'")

    # Run
    with tqdm(benchmarks, position=0) as iterator:
        for benchmark in iterator:
            iterator.set_description(f"Timing {benchmark.name}")

            run_benchmark(
                benchmark=benchmark,
                batch_sizes=batch_sizes,
                prompts=prompts,
                num_warmup_iterations=num_warmup_iterations,
                num_iterations=num_iterations,
                cooldown_time_fraction=cooldown_time_fraction,
                dtypes=dtypes,
                run_torch_cpu=run_torch_cpu,
                run_torch_mps=run_torch_mps,
                run_torch_cuda=run_torch_cuda,
                run_mlx_cpu=run_mlx_cpu,
                run_mlx_metal=run_mlx_metal,
                output_path=output_path,
            )

    print(f"Saved measurements to '{output_path}'")
    return


if __name__ == "__main__":
    fire.Fire(main)
