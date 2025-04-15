import gc
from typing import Any, Dict

import mlx.core as mx
import torch
from transformers import BatchEncoding

from mtb.hf_utils import set_hf_home


class BaseLLMBenchmark:
    """Base class for LLM benchmarks.

    Each benchmark should implement setup and run four functions:

      1. `setup_torch`
      2. `setup_mlx`
      3. `get_model_input`
      4. `run_torch`
      5. `run_mlx`:

    We can then call the following three functions in order:

      1. `setup`: Initialize for the given framework, backend, and dtype. Set up model input.
      2. `run_once`: Run the benchmark once. Of course, we can run this more than once.
      3. `teardown`: Cleanup.

    """

    def __init__(
        self,
        name: str,
        batch_size: int,
        prompt_string: str,
        max_num_tokens: int,
    ):
        self.name = name
        self.batch_size = batch_size
        self.prompt_string = prompt_string
        self.max_num_tokens = max_num_tokens

        assert batch_size == 1, "mlx_lm only supports batch-size 1 generation currently"

        self.input_tensor = None
        self.model = None
        self.tokenizer = None

        set_hf_home()

    @property
    def num_prompt_tokens(self):
        if self.model_input is None:
            raise ValueError("Model input not set. Call setup() first!")
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
            self._dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[dtype]

            torch.manual_seed(0)
            torch.set_default_device(self._device)
            torch.set_default_dtype(self._dtype)

            self.setup_torch()

        elif framework == "mlx":
            self._device = {
                "cpu": mx.Device(mx.DeviceType.cpu),
                "metal": mx.Device(mx.DeviceType.gpu),
            }[backend]

            self._dtype = {
                "float16": mx.float16,
                "bfloat16": mx.bfloat16,
                "float32": mx.float32,
            }[dtype]

            mx.random.seed(0)
            mx.set_default_device(self._device)

            self.setup_mlx()

        else:
            raise NotImplementedError(f"Unknown framework {framework}")

        # Set up model input (e.g. a prompt) for the given framework
        tensor_type = {
            "torch": "pt",
            "mlx": "mlx",
        }[framework]

        self.model_input: BatchEncoding = self.get_model_input(tensor_type=tensor_type)
        if framework == "torch":
            self.model_input.to(self._device)

    def setup_torch(self):
        raise NotImplementedError

    def setup_mlx(self):
        raise NotImplementedError

    def run_once(self) -> Dict[str, Any]:
        if self._framework == "torch":
            to_return = self.run_torch()
        elif self._framework == "mlx":
            to_return = self.run_mlx()
        elif self._framework is None:
            raise ValueError("Framework not set. Call setup() first!")
        else:
            raise NotImplementedError(f"Unknown framework {self._framework}")

        return to_return

    def run_torch(self) -> Dict[str, Any]:
        """Run the torch benchmark once."""
        raise NotImplementedError

    def run_mlx(self) -> Dict[str, Any]:
        """Run the mlx benchmark once."""
        raise NotImplementedError

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
