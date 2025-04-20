import mlx.core as mx
import torch
from transformers import BatchEncoding

from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark

__all__ = [
    "GemmaBenchmark",
    "Gemma3_1B_it_Benchmark",
    "Gemma3_4B_it_Benchmark",
]


class GemmaBenchmark(BaseLLMBenchmark):
    """Benchmarks for the Gemma class of models.

    See https://huggingface.co/collections/mlx-community/gemma-3-67d14a10480a436ad478b0f9
    for available models.

    """

    def set_prompt(self, prompt: str, batch_size: int) -> BatchEncoding:
        assert batch_size == 1, "mlx_lm only supports B=1 inference currently "

        # Input-finetuned models ('-pi' suffix) expect a format that specifies input type
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ]

        tensor_type = {
            "torch": "pt",
            "mlx": "mlx",
        }[self._framework]

        self.model_input = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=tensor_type,
        )
        return self.model_input


class Gemma3_1B_it_Benchmark(GemmaBenchmark):
    dtype_to_model_id = {
        torch.bfloat16: "google/gemma-3-1b-it",
        mx.bfloat16: "mlx-community/gemma-3-1b-it-bf16",
        mx.int8: "mlx-community/gemma-3-1b-it-8bit",
        "mx_int4": "mlx-community/gemma-3-1b-pt-4bit",
    }
    name = "gemma-3-1b-it"


class Gemma3_4B_it_Benchmark(GemmaBenchmark):
    dtype_to_model_id = {
        torch.bfloat16: "google/gemma-3-4b-it",
        mx.bfloat16: "mlx-community/gemma-3-4b-it-bf16",
        mx.int8: "mlx-community/gemma-3-4b-it-8bit",
        "mx.int4": "mlx-community/gemma-3-4b-it-4bit",
    }
    name = "gemma-3-4b-it"
