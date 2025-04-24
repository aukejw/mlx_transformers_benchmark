import gc
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark
from mtb.measurement import LlmBenchmarkMeasurement


class TorchLlmBenchmark(BaseLLMBenchmark):
    """Base class for LLM benchmarks in torch."""

    framework = "torch"

    def setup(self):
        self._device = {
            "cpu": torch.device("cpu"),
            "mps": torch.device("mps"),
            "cuda": torch.device("cuda"),
        }[self.backend]

        torch.set_default_device(self._device)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self._device,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
        )

    def format_prompt(self, prompt: str) -> torch.Tensor:
        prompt = self.prompt_formatter(prompt)

        model_input = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return model_input.input_ids

    @torch.inference_mode()
    def run_once(self, prompt: Any) -> LlmBenchmarkMeasurement:
        """Run the torch benchmark once. Return measurements.

        Note that we approximate time-to-first-token by measuring the time
        until the end of the first inference pass. This is not exact, but
        should be close enough for our purposes.

        """
        prompt_tokens = self.format_prompt(prompt)
        num_prompt_tokens = prompt_tokens.shape[1]

        # Time processing of prompt, initializing kv cache via a forward hook
        # for the end of the first forward pass.
        stats_after_first_forward = None

        def log_time_hook(module, input, output):
            nonlocal stats_after_first_forward
            if stats_after_first_forward is None:
                stats_after_first_forward = dict(end_time=time.time_ns())

        hook_handle = self.model.register_forward_hook(log_time_hook)

        # Generate tokens (from scratch, no previous kv cache)
        start_time = time.time_ns()
        generation: torch.Tensor = self.model.generate(
            input_ids=prompt_tokens,
            max_new_tokens=self.max_num_tokens,
            do_sample=False,
            top_k=None,
            top_p=None,
        )
        num_generated_tokens = generation.shape[1] - num_prompt_tokens
        generation = self.tokenizer.batch_decode(generation[0, num_prompt_tokens:])
        generation = "".join(generation)

        # Collect metrics. Assumption: total_time - ttft = generation_time
        end_time = time.time_ns()

        prompt_seconds = (stats_after_first_forward["end_time"] - start_time) / 1e9
        prompt_tps = num_prompt_tokens / prompt_seconds
        generation_seconds = (end_time - start_time) / 1e9 - prompt_seconds
        generation_tps = num_generated_tokens / generation_seconds

        hook_handle.remove()

        return LlmBenchmarkMeasurement(
            response=generation,
            prompt_tps=prompt_tps,
            prompt_time_sec=prompt_seconds,
            generation_tps=generation_tps,
            generation_time_sec=generation_seconds,
            num_prompt_tokens=num_prompt_tokens,
            num_generated_tokens=num_generated_tokens,
        )

    def teardown(self):
        """Teardown the benchmark."""

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None

        if self._device.type == "mps":
            torch.mps.empty_cache()
        elif self._device.type == "cuda":
            torch.cuda.empty_cache()

        gc.collect()
