import time
from typing import Any, Dict

import mlx.core as mx
import mlx.nn
import mlx_lm
import mlx_lm.tokenizer_utils
import torch
from transformers import AutoTokenizer, BatchEncoding, Gemma3ForCausalLM

from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark

__all__ = [
    "GemmaBenchmark",
    "Gemma3OneBillionBenchmark",
    "Gemma3FourBillionBenchmark",
]


class GemmaBenchmark(BaseLLMBenchmark):
    model_name: str = None

    def __init__(
        self,
        max_num_tokens: int = 100,
    ):
        name = f"GemmaBenchmark({self.model_name})"

        super().__init__(
            name=name,
            max_num_tokens=max_num_tokens,
        )

    def setup_torch(self):
        model_id = self.dtype_to_model_id[self._dtype]

        self.model = Gemma3ForCausalLM.from_pretrained(
            model_id,
            device_map=self._device,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def setup_mlx(self):
        model_id = self.dtype_to_model_id[self._dtype]

        model, tokenizer = mlx_lm.load(model_id)
        self.model: mlx.nn.Module = model
        self.tokenizer: mlx_lm.tokenizer_utils.TokenizerWrapper = tokenizer

    def set_prompt(self, prompt: str, batch_size: int) -> BatchEncoding:
        assert batch_size == 1, "mlx_lm only supports B=1 inference currently "

        # Input-finetuned models ('-pi' suffix) expect a specific format
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

    @torch.inference_mode()
    def run_torch_generate(self) -> Dict[str, Any]:
        input_ids = self.model_input["input_ids"]
        num_prompt_tokens = input_ids.shape[1]

        # Time processing of prompt, initializing kv cache via a forward hook
        # for the very first forward pass.
        stats_after_first_forward = dict()
        memory_tracker = self.memory_tracker

        def log_memory_hook(module, input, output):
            nonlocal stats_after_first_forward, memory_tracker
            if not len(stats_after_first_forward):
                stats_after_first_forward = dict(
                    ram_used_gb=memory_tracker.get_used_memory(),
                    end_time=time.time_ns(),
                )

        self.model.register_forward_hook(log_memory_hook)

        # Generate tokens (from scratch, no previous kv cache)
        start_time = time.time_ns()
        generation: torch.Tensor = self.model.generate(
            **self.model_input,
            max_new_tokens=self.max_num_tokens,
            do_sample=False,
            top_k=None,
            top_p=None,
        )
        num_generated_tokens = generation.shape[1] - num_prompt_tokens
        generation = self.tokenizer.batch_decode(generation[0, num_prompt_tokens:])
        generation = "".join(generation)

        # collect metrics
        end_time = time.time_ns()
        prompt_seconds = (stats_after_first_forward["end_time"] - start_time) / 1e9
        generation_seconds = (end_time - start_time) / 1e9 - prompt_seconds

        prompt_tps = num_prompt_tokens / prompt_seconds
        generation_tps = num_generated_tokens / generation_seconds

        memory_gb = stats_after_first_forward["ram_used_gb"]

        return dict(
            generation=generation,
            prompt_tps=prompt_tps,
            prompt_time_sec=prompt_seconds,
            num_generated_tokens=num_generated_tokens,
            generation_tps=generation_tps,
            ram_used_gb=memory_gb,
        )

    def run_mlx_generate(self) -> Dict[str, Any]:
        # TODO mlx_lm generation only supports a single prompt, not batch-of-prompt (2025-04-15)
        assert self.model_input["input_ids"].shape[0] == 1, "mlx_lm only supports B=1"
        prompt = self.model_input["input_ids"][0]

        # track stats after first forward, when memory pressure is highest (kv cache init)
        stats_after_first_forward = None

        # use stream_generate instead of generate, its response is more useful
        generation = ""
        for response in mlx_lm.stream_generate(
            self.model,
            self.tokenizer,
            max_tokens=self.max_num_tokens,
            prompt=prompt,
        ):
            generation += response.text

            if stats_after_first_forward is None:
                stats_after_first_forward = dict(
                    ram_used_gb=self.memory_tracker.get_used_memory(),
                )

        return dict(
            generation=generation,
            prompt_tps=response.prompt_tps,
            prompt_time_sec=response.prompt_tokens / response.prompt_tps,
            num_generated_tokens=response.generation_tokens,
            generation_tps=response.generation_tps,
            peak_memory_gb=response.peak_memory,
            ram_used_gb=stats_after_first_forward["ram_used_gb"],
        )


class Gemma3OneBillionBenchmark(GemmaBenchmark):
    dtype_to_model_id = {
        torch.bfloat16: "google/gemma-3-1b-it",
        mx.bfloat16: "mlx-community/gemma-3-1b-it-bf16",
    }
    model_name = "gemma-3-1b-it"


class Gemma3FourBillionBenchmark(GemmaBenchmark):
    dtype_to_model_id = {
        torch.bfloat16: "google/gemma-3-4b-it",
        mx.bfloat16: "mlx-community/gemma-3-4b-it-bf16",
    }
    model_name = "gemma-3-4b-it"
