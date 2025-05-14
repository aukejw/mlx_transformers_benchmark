from typing import Any

from mtb.llm_benchmarks.models.base import ModelSpec


def format_deepseek_prompt(prompt: str) -> Any:
    """Deepseek models expect a regular system prompt."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return messages


Deepseek_R1_Distill_Qwen_7B = ModelSpec(
    name="Deepseek-R1-Distill-7B",
    num_params=int(7e9),
    prompt_formatter=format_deepseek_prompt,
    model_ids={
        "mlx": {
            "int4": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        },
        "lmstudio": {
            "int4": "lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        },
        "ollama": {
            "int4": "deepseek-r1:7b",
        },
    },
)
