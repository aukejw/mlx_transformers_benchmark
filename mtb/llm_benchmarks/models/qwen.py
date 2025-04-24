from dataclasses import dataclass
from typing import Any

from mtb.llm_benchmarks.models.base import ModelSpec

__all__ = [
    "QwenPromptFormatter",
    "Qwen2p5_0p5B_it",
    "Qwen2p5_Coder_0p5B_it",
    "Qwen2p5_3B_it",
    "Qwen2p5_Coder_3B_it",
]


class QwenPromptFormatter:
    """Input-finetuned models ('-pi' suffix) expect a specific format."""

    def __call__(self, prompt: str) -> Any:
        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        return messages


@dataclass
class Qwen2p5_0p5B_it(ModelSpec):
    name = "qwen-2.5-0.5b-it"
    num_params = 5e8
    prompt_formatter = QwenPromptFormatter

    model_ids = {
        "torch": {
            "bfloat16": "Qwen/Qwen2.5-0.5B-Instruct",
        },
        "mlx": {
            "bfloat16": "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
            "int8": "mlx-community/Qwen2.5-0.5B-Instruct-8bit",
            "int4": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        },
    }


@dataclass
class Qwen2p5_3B_it(ModelSpec):
    name = "qwen-2.5-3b-it"
    num_params = 3e9
    prompt_formatter = QwenPromptFormatter

    model_ids = {
        "torch": {
            "bfloat16": "Qwen/Qwen2.5-3B-Instruct",
        },
        "mlx": {
            "bfloat16": "mlx-community/Qwen2.5-3B-Instruct-bf16",
            "int8": "mlx-community/Qwen2.5-3B-Instruct-8bit",
            "int4": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        },
    }


@dataclass
class Qwen2p5_Coder_0p5B_it(ModelSpec):
    name = "qwen-2.5-coder-0.5b-it"
    num_params = 5e8
    prompt_formatter = QwenPromptFormatter

    model_ids = {
        "torch": {
            "bfloat16": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        },
        "mlx": {
            "bfloat16": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-bf16",
            "int8": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-8bit",
            "int4": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
        },
    }


@dataclass
class Qwen2p5_Coder_3B_it(ModelSpec):
    name = "qwen-2.5-coder-3b-it"
    num_params = 3e9
    prompt_formatter = QwenPromptFormatter

    model_ids = {
        "torch": {
            "bfloat16": "Qwen/Qwen2.5-Coder-3B-Instruct",
        },
        "mlx": {
            "bfloat16": "mlx-community/Qwen2.5-Coder-3B-Instruct-bf16",
            "int8": "mlx-community/Qwen2.5-Coder-3B-Instruct-8bit",
            "int4": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
        },
    }
