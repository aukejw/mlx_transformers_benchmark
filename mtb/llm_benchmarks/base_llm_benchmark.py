import gc
import time
from typing import Any, Dict

import mlx.core as mx
import mlx.nn
import mlx_lm
import mlx_lm.tokenizer_utils
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mtb.dtypes import get_mlx_dtype, get_torch_dtype
from mtb.hf_utils import get_hf_home, verbose_download_model
from mtb.measurement import LlmBenchmarkMeasurement


class BaseLLMBenchmark:
    """Base class for LLM benchmarks.

    Each benchmark should implement at least prompt creation:

      1. `set_prompt`

    Optionally, we can override setup and run functions:

      1. `setup_torch`
      2. `setup_mlx`
      3. `run_torch_generate`
      4. `run_mlx_generate`:

    We can then call the following three functions in order:

      1. `setup`: Initialize for the given framework, backend, and dtype. Set up model input.
      2. `run_once`: Run the benchmark once. Of course, we can run this more than once.
      3. `teardown`: Cleanup.

    """

    # mapping from dtype to model_id
    dtype_to_model_id: Dict[Any, str] = {}
    # benchmark identifier
    name: str = None

    def __init__(
        self,
        max_num_tokens: int = 100,
    ):
        self.max_num_tokens = max_num_tokens

        self.input_tensor = None
        self.model = None
        self.tokenizer = None

    @property
    def num_prompt_tokens(self):
        if self.model_input is None:
            raise ValueError("Model input not set. Call set_prompt() first!")
        return self.model_input["input_ids"].shape[1]

    def setup(
        self,
        framework: str,
        backend: str,
        dtype: str,
    ):
        """Setup the benchmark."""
        self._framework = framework
        self._backend = backend

        if framework == "torch":
            self._device = torch.device(backend)
            self._dtype = get_torch_dtype(dtype)

            if self._dtype not in self.dtype_to_model_id:
                raise ValueError(
                    f"For benchmark '{self.name}', we do not know the model for dtype {self._dtype}"
                )

            torch.manual_seed(0)
            torch.set_default_device(self._device)
            torch.set_default_dtype(self._dtype)

            self.setup_torch()

        elif framework == "mlx":
            self._device = {
                "cpu": mx.Device(mx.DeviceType.cpu),
                "metal": mx.Device(mx.DeviceType.gpu),
            }[backend]

            self._dtype = get_mlx_dtype(dtype)

            if self._dtype not in self.dtype_to_model_id:
                raise ValueError(
                    f"For benchmark '{self.name}', we do not know the model for dtype {self._dtype}"
                )

            mx.random.seed(0)
            mx.set_default_device(self._device)

            self.setup_mlx()

        else:
            raise NotImplementedError(f"Unknown framework {framework}")

    def setup_torch(self):
        model_id = self.dtype_to_model_id[self._dtype]
        verbose_download_model(model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=self._device,
            cache_dir=get_hf_home(),
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=get_hf_home(),
        )

    def setup_mlx(self):
        model_id = self.dtype_to_model_id[self._dtype]
        verbose_download_model(model_id)

        model, tokenizer = mlx_lm.load(model_id)
        self.model: mlx.nn.Module = model
        self.tokenizer: mlx_lm.tokenizer_utils.TokenizerWrapper = tokenizer

    def set_prompt(self, prompt: str, batch_size: int):
        """Set the model_input attribute for this benchmark."""
        raise NotImplementedError

    def run_once(self) -> LlmBenchmarkMeasurement:
        if self._framework == "torch":
            to_return = self.run_torch_generate()
        elif self._framework == "mlx":
            to_return = self.run_mlx_generate()
        elif self._framework is None:
            raise ValueError("Framework not set. Call setup() first!")
        else:
            raise NotImplementedError(f"Unknown framework {self._framework}")

        return to_return

    @torch.inference_mode()
    def run_torch_generate(self) -> LlmBenchmarkMeasurement:
        """Run the torch benchmark once. Return measurements."""
        input_ids = self.model_input["input_ids"]
        num_prompt_tokens = input_ids.shape[1]

        # Time processing of prompt, initializing kv cache via a forward hook
        # for the very first forward pass.
        stats_after_first_forward = None

        def log_time_hook(module, input, output):
            nonlocal stats_after_first_forward
            if stats_after_first_forward is None:
                stats_after_first_forward = dict(end_time=time.time_ns())

        self.model.register_forward_hook(log_time_hook)

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
        prompt_tps = num_prompt_tokens / prompt_seconds
        generation_seconds = (end_time - start_time) / 1e9 - prompt_seconds
        generation_tps = num_generated_tokens / generation_seconds

        return LlmBenchmarkMeasurement(
            response=generation,
            prompt_tps=prompt_tps,
            prompt_time_sec=prompt_seconds,
            generation_tps=generation_tps,
            generation_time_sec=generation_seconds,
            num_prompt_tokens=num_prompt_tokens,
            num_generated_tokens=num_generated_tokens,
        )

    def run_mlx_generate(self) -> LlmBenchmarkMeasurement:
        """Run the mlx benchmark once. Return measurements."""
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

        return LlmBenchmarkMeasurement(
            response=generation,
            prompt_tps=response.prompt_tps,
            prompt_time_sec=response.prompt_tokens / response.prompt_tps,
            generation_time_sec=response.generation_tokens / response.generation_tps,
            generation_tps=response.generation_tps,
            num_prompt_tokens=response.prompt_tokens,
            num_generated_tokens=response.generation_tokens,
        )

    def teardown(self):
        """Teardown the benchmark."""
        del self.model
        del self.tokenizer
        del self.input_tensor

        self.model = None
        self.tokenizer = None
        self.input_tensor = None

        if self._framework == "torch":
            if self._device.type == "mps":
                torch.mps.empty_cache()
            elif self._device.type == "cuda":
                torch.cuda.empty_cache()

        elif self._framework == "mlx":
            mx.clear_cache()

        # Reset indicators
        self._framework = None
        self._backend = None
        self._dtype = None
        self._device = None

        gc.collect()
        return
