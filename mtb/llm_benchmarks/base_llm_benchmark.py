from typing import Any, Callable

from mtb.measurement import LlmBenchmarkMeasurement


class BaseLLMBenchmark:
    """Base class for LLM benchmarks.

    Should implement:

      1. `setup`: Initialize the model and tokenizer. Prepare the prompt.
      2. `run_once`: Run the benchmark once.
      3. `teardown`: Cleanup.

    """

    # Name of the framework used for the benchmark
    framework = None

    def __init__(
        self,
        name: str,
        model_id: str,
        backend: str,
        dtype: str,
        prompt_formatter: Callable[[str], Any],
        max_num_tokens: int = 100,
    ):
        self.name = name
        self.model_id = model_id
        self.backend = backend
        self.dtype = dtype
        self.prompt_formatter = prompt_formatter
        self.max_num_tokens = max_num_tokens

    def format_prompt(self, prompt: str) -> Any:
        """Format the given prompt"""
        raise NotImplementedError

    def get_num_prompt_tokens(self, prompt: str) -> int:
        """Get the number of tokens in the prompt."""
        tokens = self.format_prompt(prompt)
        return len(tokens[0])

    def setup(self):
        """Set up the benchmark. Load the model, tokenizer."""
        raise NotImplementedError

    def run_once(self) -> LlmBenchmarkMeasurement:
        """Run the benchmark once. Return measurements."""
        raise NotImplementedError

    def teardown(self):
        """Teardown the benchmark."""
        raise NotImplementedError
