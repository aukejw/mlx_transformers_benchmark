import gc
from typing import Any, List

import lmstudio as lms
import numpy as np

from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark
from mtb.lmstudio_utils import check_lms_server_running
from mtb.measurement import LlmBenchmarkMeasurement


class LMStudioLlmBenchmark(BaseLLMBenchmark):
    """Base class for LLM benchmarks in LM Studio."""

    framework = "lmstudio"

    def __init__(
        self,
        max_context_length: int = 5000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_context_length = max_context_length

        if not check_lms_server_running():
            raise ValueError("LM Studio server is not running!")

    def setup(self):
        """Set up the benchmark. Load the model, tokenizer."""
        self.model = lms.llm(
            self.model_id,
            config=dict(
                contextLength=self.max_context_length,
            ),
        )
        return

    def format_prompt(self, prompt: str) -> List[List[int]]:
        """Format the given prompt."""
        prompt = self.prompt_formatter(prompt)
        if isinstance(prompt, list):
            prompt = {"messages": prompt}
        prompt = self.model.apply_prompt_template(prompt)
        prompt_tokens = self.model.tokenize(prompt)
        return np.array([prompt_tokens])

    def run_once(self, prompt: Any) -> LlmBenchmarkMeasurement:
        """Run the benchmark once. Return measurements."""
        response: lms.json_api.PredictionResult = self.model.respond(
            prompt,
            config=dict(
                temperature=0.0,
                maxTokens=self.max_num_tokens,
            ),
        )
        stats = response.stats

        return LlmBenchmarkMeasurement(
            response=response.content,
            prompt_time_sec=stats.time_to_first_token_sec,
            prompt_tps=stats.prompt_tokens_count / stats.time_to_first_token_sec,
            generation_time_sec=stats.predicted_tokens_count / stats.tokens_per_second,
            generation_tps=stats.tokens_per_second,
            num_prompt_tokens=stats.prompt_tokens_count,
            num_generated_tokens=stats.predicted_tokens_count,
        )

    def teardown(self):
        """Teardown the benchmark."""

        # Unload, delete references
        self.model.unload()
        del self.model
        self.model = None

        # Reset indicators
        self._backend = None
        self._dtype = None
        self._device = None

        gc.collect()
        return
