import mlx.core as mx
import torch
from transformers import BatchEncoding

from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark

__all__ = [
    "QwenBenchmark",
    "Qwen2p5_3B_it_Benchmark",
    "Qwen2p5_Coder_3B_it_Benchmark",
]


class QwenBenchmark(BaseLLMBenchmark):
    def set_prompt(self, prompt: str, batch_size: int) -> BatchEncoding:
        assert batch_size == 1, "mlx_lm only supports B=1 inference currently "

        # Input-finetuned models ('-pi' suffix) expect a specific format
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


class Qwen2p5_3B_it_Benchmark(QwenBenchmark):
    name = "qwen-2.5-3b-it"

    dtype_to_model_id = {
        torch.bfloat16: "Qwen/Qwen2.5-3B-Instruct",
        mx.bfloat16: "mlx-community/Qwen2.5-3B-Instruct-bf16",
    }


class Qwen2p5_Coder_3B_it_Benchmark(QwenBenchmark):
    name = "qwen-2.5-coder-3b-it"

    dtype_to_model_id = {
        torch.bfloat16: "Qwen/Qwen2.5-Coder-3B-Instruct",
        mx.bfloat16: "mlx-community/Qwen2.5-Coder-3B-Instruct-bf16",
    }
