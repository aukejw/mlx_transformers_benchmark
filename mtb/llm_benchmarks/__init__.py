from mtb.llm_benchmarks.gemma import Gemma3_1B_it_Benchmark, Gemma3_4B_it_Benchmark
from mtb.llm_benchmarks.qwen import (
    Qwen2p5_0p5B_it_Benchmark,
    Qwen2p5_3B_it_Benchmark,
    Qwen2p5_Coder_0p5B_it_Benchmark,
    Qwen2p5_Coder_3B_it_Benchmark,
)

__all__ = [
    # Gemma 3 models
    "Gemma3_1B_it_Benchmark",
    "Gemma3_4B_it_Benchmark",
    # Qwen 2.5 models
    "Qwen2p5_0p5B_it_Benchmark",
    "Qwen2p5_Coder_0p5B_it_Benchmark",
    "Qwen2p5_3B_it_Benchmark",
    "Qwen2p5_Coder_3B_it_Benchmark",
]
