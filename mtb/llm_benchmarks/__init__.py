from mtb.llm_benchmarks.models.deepseek import (
    Deepseek_R1_0528_Qwen3_8B,
    Deepseek_R1_Distill_Qwen_7B,
)
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
    Qwen3_0p6B_it,
    Qwen3_8B_it,
    Qwen3_14B_it,
)

MODEL_SPECS = [
    # deepseek
    Deepseek_R1_Distill_Qwen_7B,
    Deepseek_R1_0528_Qwen3_8B,
    # gemma
    Gemma3_1B_it,
    Gemma3_1B_it_QAT,
    Gemma3_4B_it,
    Gemma3_4B_it_QAT,
    Gemma3_12B_it_QAT,
    # qwen
    Qwen2p5_0p5B_it,
    Qwen2p5_3B_it,
    Qwen2p5_Coder_0p5B_it,
    Qwen2p5_Coder_3B_it,
    Qwen3_0p6B_it,
    Qwen3_8B_it,
    Qwen3_14B_it,
]
