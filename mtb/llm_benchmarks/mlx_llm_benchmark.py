import gc

import mlx
import mlx.core as mx
import mlx_lm
import mlx_lm.tokenizer_utils

from mtb.hf_utils import verbose_download_model
from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark
from mtb.measurement import LlmBenchmarkMeasurement


class MlxLlmBenchmark(BaseLLMBenchmark):
    """Base class for LLM benchmarks in mlx."""

    framework = "mlx"

    def setup(self):
        """Set up the benchmark. Load the model, tokenizer."""
        self._device = {
            "cpu": mx.Device(mx.DeviceType.cpu),
            "metal": mx.Device(mx.DeviceType.gpu),
        }[self.backend]

        mx.set_default_device(self._device)

        verbose_download_model(self.model_id)
        model, tokenizer = mlx_lm.load(self.model_id)
        self.model: mlx.nn.Module = model
        self.tokenizer: mlx_lm.tokenizer_utils.TokenizerWrapper = tokenizer

    def format_prompt(self, prompt: str) -> mx.array:
        prompt = self.prompt_formatter(prompt)

        model_input = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="mlx",
        )
        return model_input.input_ids

    def run_once(self, prompt: str) -> LlmBenchmarkMeasurement:
        """Run the mlx benchmark once. Return measurements."""
        mx.reset_peak_memory()
        prompt_tokens = self.format_prompt(prompt)

        # mlx only accepts B=1
        prompt_tokens = prompt_tokens[0]

        # use stream_generate instead of generate, its response is more useful
        generation = ""
        for response in mlx_lm.stream_generate(
            self.model,
            self.tokenizer,
            max_tokens=self.max_num_tokens,
            prompt=prompt_tokens,
        ):
            generation += response.text

        return LlmBenchmarkMeasurement(
            response=generation,
            prompt_tps=response.prompt_tps,
            prompt_time_sec=response.prompt_tokens / response.prompt_tps,
            generation_time_sec=response.generation_tokens / response.generation_tps,
            generation_tps=response.generation_tps,
            num_prompt_tokens=response.prompt_tokens,
            num_generated_tokens=response.generation_tokens,
            peak_memory_gib=response.peak_memory,
        )

    def teardown(self):
        """Teardown the benchmark."""

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None

        mx.clear_cache()
        gc.collect()
