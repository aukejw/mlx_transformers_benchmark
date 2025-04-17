import gc
from typing import Any, Dict

import mlx.core as mx
import torch

from mtb.dtypes import get_mlx_dtype, get_torch_dtype
from mtb.hf_utils import set_hf_home
from mtb.memory import MemoryTracker


class BaseLLMBenchmark:
    """Base class for LLM benchmarks.

    Each benchmark should implement setup and run four functions:

      1. `setup_torch`
      2. `setup_mlx`
      3. `set_prompt`
      4. `run_torch_generate`
      5. `run_mlx_generate`:

    We can then call the following three functions in order:

      1. `setup`: Initialize for the given framework, backend, and dtype. Set up model input.
      2. `run_once`: Run the benchmark once. Of course, we can run this more than once.
      3. `teardown`: Cleanup.

    """

    dtype_to_model_id = dict()

    def __init__(
        self,
        name: str,
        max_num_tokens: int,
    ):
        self.name = name
        self.max_num_tokens = max_num_tokens

        self.input_tensor = None
        self.model = None
        self.tokenizer = None
        self.memory_tracker = None

        set_hf_home()

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

        self.memory_tracker = MemoryTracker(framework=framework, backend=backend)

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
        raise NotImplementedError

    def setup_mlx(self):
        raise NotImplementedError

    def set_prompt(self, prompt: str, batch_size: int):
        """Set the model_input attribute for this benchmark."""
        raise NotImplementedError

    def run_once(self) -> Dict[str, Any]:
        if self._framework == "torch":
            to_return = self.run_torch_generate()
        elif self._framework == "mlx":
            to_return = self.run_mlx_generate()
        elif self._framework is None:
            raise ValueError("Framework not set. Call setup() first!")
        else:
            raise NotImplementedError(f"Unknown framework {self._framework}")

        return to_return

    def run_torch_generate(self) -> Dict[str, Any]:
        """Run the torch benchmark once."""
        raise NotImplementedError

    def run_mlx_generate(self) -> Dict[str, Any]:
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
