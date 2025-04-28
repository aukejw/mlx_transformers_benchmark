from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import fire
from tqdm import tqdm

import mtb as mtb
from mtb.file_io import create_benchmark_output_dir
from mtb.llm_benchmarks.models.deepseek import Deepseek_R1_Distill_Qwen_7B
from mtb.llm_benchmarks.models.gemma import (
    Gemma3_1B_it,
    Gemma3_1B_it_QAT,
    Gemma3_4B_it,
    Gemma3_4B_it_QAT,
    Gemma3_12B_it_QAT,
)
from mtb.llm_benchmarks.models.qwen import (
    Qwen2p5_0p5B_it,
    Qwen2p5_3B_it,
    Qwen2p5_Coder_0p5B_it,
    Qwen2p5_Coder_3B_it,
)
from mtb.run_llm_benchmark import run_benchmark
from mtb.select_benchmarks import filter_llm_benchmarks

DEFAULT_OUTPUT_ROOT = mtb.REPO_ROOT / "measurements" / "llm_benchmarks"


def main(
    output_root: Union[str, Path] = DEFAULT_OUTPUT_ROOT,
    num_warmup_iterations: int = 1,
    num_iterations: int = 5,
    batch_sizes: Iterable[int] = (1,),
    dtypes: str = ("int4", "int8", "bfloat16"),
    prompt_lengths: Iterable[int] = (4096, 1024, 256, 64),
    max_num_tokens: int = 100,
    enable_hf_progressbar: bool = False,
    cooldown_time_fraction: float = 0.05,
    *,
    run_only_benchmarks: Optional[Iterable[str]] = None,
    run_mlx_metal: bool = True,
    run_torch_mps: bool = False,
    run_torch_cpu: bool = False,
    run_torch_cuda: bool = False,
    run_mlx_cpu: bool = False,
    run_lmstudio_metal: bool = False,
):
    """Run LLM benchmarks.

    To avoid frying the hardware, we add a small cooldown during which the chips
    should (mostly) idle. A cooldown of 10% of the duration of the task results
    in a 95°C peak GPU temperature on a Macbook M4 Pro, but YMMV.

    By default, we run MLX with Metal backend only.

    """
    from mtb.hf_utils import set_hf_home

    set_hf_home(enable_hf_progressbar=enable_hf_progressbar)

    boolean_flags = dict(
        run_mlx_metal=run_mlx_metal,
        run_torch_mps=run_torch_mps,
        run_torch_cpu=run_torch_cpu,
        run_torch_cuda=run_torch_cuda,
        run_mlx_cpu=run_mlx_cpu,
        run_lmstudio_metal=run_lmstudio_metal,
    )

    # Define the model specs to benchmark
    model_specs = [
        # gemma 3
        Gemma3_1B_it,
        Gemma3_4B_it,
        # gemma 3 qat
        Gemma3_1B_it_QAT,
        Gemma3_4B_it_QAT,
        Gemma3_12B_it_QAT,
        # qwen 2.5
        Qwen2p5_0p5B_it,
        Qwen2p5_3B_it,
        # qwen 2.5 coder
        Qwen2p5_Coder_0p5B_it,
        Qwen2p5_Coder_3B_it,
        # deepseek
        Deepseek_R1_Distill_Qwen_7B,
    ]

    # Filter benchmarks if specified
    benchmarks_to_run: List[Dict] = filter_llm_benchmarks(
        model_specs=model_specs,
        dtypes=dtypes,
        run_only_benchmarks=run_only_benchmarks,
        **boolean_flags,
    )

    # Create output directory for measurements
    output_dir = create_benchmark_output_dir(
        output_root=output_root,
        benchmark_settings=dict(
            num_warmup_iterations=num_warmup_iterations,
            num_iterations=num_iterations,
            max_num_tokens=max_num_tokens,
            dtypes=dtypes,
            run_only_benchmarks=run_only_benchmarks,
        ),
    )
    output_path = output_dir / "benchmark_results.csv"
    print(f"\nOutput directory: '{output_dir}'")

    # Run
    with tqdm(benchmarks_to_run, position=0) as iterator:
        for benchmark_config in iterator:
            model_spec = benchmark_config["model_spec"]
            iterator.set_description(
                f"Benchmarking {model_spec.name}, "
                f"{benchmark_config['framework']}+{benchmark_config['backend']}, "
                f"dtype={benchmark_config['dtype']}"
            )

            measurements = run_benchmark(
                **benchmark_config,
                output_path=output_path,
                batch_sizes=batch_sizes,
                prompt_lengths=prompt_lengths,
                num_warmup_iterations=num_warmup_iterations,
                num_iterations=num_iterations,
                max_num_tokens=max_num_tokens,
                cooldown_time_fraction=cooldown_time_fraction,
            )

    print(f"Saved {len(measurements)} measurements to '{output_path}'")
    return


if __name__ == "__main__":
    fire.Fire(main)
