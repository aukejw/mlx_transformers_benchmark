from mtb.llm_benchmarks.gemma import (
    Gemma3_1B_it_Benchmark,
    Gemma3_1B_it_QAT_Benchmark,
    Gemma3_4B_it_Benchmark,
    Gemma3_4B_it_QAT_Benchmark,
    Gemma3_12B_it_QAT_Benchmark,
)
from mtb.llm_benchmarks.qwen import (
    Qwen2p5_0p5B_it_Benchmark,
    Qwen2p5_3B_it_Benchmark,
    Qwen2p5_Coder_0p5B_it_Benchmark,
    Qwen2p5_Coder_3B_it_Benchmark,
)

__all__ = [
    # Gemma 3
    "Gemma3_1B_it_Benchmark",
    "Gemma3_4B_it_Benchmark",
    # Gemma 3 qat
    "Gemma3_1B_it_QAT_Benchmark",
    "Gemma3_4B_it_QAT_Benchmark",
    "Gemma3_12B_it_QAT_Benchmark",
    # Qwen 2.5
    "Qwen2p5_0p5B_it_Benchmark",
    "Qwen2p5_3B_it_Benchmark",
    # Qwen 2.5 coder
    "Qwen2p5_Coder_0p5B_it_Benchmark",
    "Qwen2p5_Coder_3B_it_Benchmark",
]
