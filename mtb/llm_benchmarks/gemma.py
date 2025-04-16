import time
from typing import Any, Dict

import mlx.core as mx
import mlx.nn
import mlx_lm
import mlx_lm.tokenizer_utils
import torch
from transformers import AutoTokenizer, BatchEncoding, Gemma3ForCausalLM

from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark


class GemmaBenchmark(BaseLLMBenchmark):
    dtype_to_model_id = {
        torch.bfloat16: "google/gemma-3-1b-it",
        mx.bfloat16: "mlx-community/gemma-3-1b-it-bf16",
    }

    def __init__(
        self,
        max_num_tokens: int = 100,
    ):
        model_name = "gemma-3-1b-it"
        name = f"GemmaBenchmark({model_name})"

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

        # Time processing of prompt, initializing kv cache.
        # It is challenging to handle inference with kv caching in a general way,
        # as each model handles position indices and cache indices differently.
        # Instead, we time the model processing the prompt once, and call generate later.
        tic = time.perf_counter()
        outputs = self.model(**self.model_input, use_cache=True)

        if self._backend == "cuda":
            torch.cuda.synchronize()
        elif self._backend == "mps":
            torch.mps.synchronize()

        prompt_time_sec = time.perf_counter() - tic
        prompt_tps = num_prompt_tokens / prompt_time_sec
        del outputs

        # Generate tokens (from scratch, no previous kv cache)
        tic = time.perf_counter()
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

        generation_seconds = time.perf_counter() - tic
        generation_seconds -= prompt_time_sec
        generation_tps = self.max_num_tokens / generation_seconds

        return dict(
            generation=generation,
            prompt_tps=prompt_tps,
            prompt_time_sec=prompt_time_sec,
            num_generated_tokens=num_generated_tokens,
            generation_tps=generation_tps,
            current_memory_gb=torch.mps.current_allocated_memory() / 1024**3,
        )

    def run_mlx_generate(self) -> Dict[str, Any]:
        # TODO mlx_lm generation only supports a single prompt, not batch-of-prompt (2025-04-15)
        assert self.model_input["input_ids"].shape[0] == 1, "mlx_lm only supports B=1"
        prompt = self.model_input["input_ids"][0]

        # use stream_generate instead of generate, its response is more useful
        generation = ""
        for response in mlx_lm.stream_generate(
            self.model,
            self.tokenizer,
            max_tokens=self.max_num_tokens,
            prompt=prompt,
        ):
            generation += response.text

        return dict(
            generation=generation,
            prompt_tps=response.prompt_tps,
            prompt_time_sec=response.prompt_tokens / response.prompt_tps,
            num_generated_tokens=response.generation_tokens,
            generation_tps=response.generation_tps,
            peak_memory_gb=response.peak_memory,
            current_memory_gb=mx.get_active_memory() / 1024**3,
        )
