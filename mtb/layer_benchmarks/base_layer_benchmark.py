import gc
from typing import Tuple

import mlx.core as mx
import torch


class BaseLayerBenchmark:
    """Benchmark class containing entrypoint functions.

    Each benchmark should implement setup and run four functions:

      1. `setup_torch`
      2. `setup_mlx`
      3. `run_torch`
      4. `run_mlx`:

    We can then call the following three functions in order:

      1. `setup`: Initialize for the given framework, backend, and dtype.
      2. `run_once`: Run the benchmark once. Of course, we can run this more than once.
      3. `teardown`: Cleanup.

    """

    def __init__(
        self,
        name: str,
        input_shape: Tuple[int, int, int],
    ):
        self.name = name
        self.input_shape = input_shape

        self._batch_size = input_shape[0]
        self._num_tokens = input_shape[1]
        self._num_features = input_shape[2]

        # placeholders set during setup
        self._framework = None
        self._backend = None
        self._compile = None
        self._device = None
        self._dtype = None

        # placeholders for the function and tensors
        self.torch_function = None
        self.mlx_function = None
        self.input_tensor = None

    def setup_torch(self):
        """Setup the torch benchmark."""
        raise NotImplementedError

    def setup_mlx(self):
        """Setup the mlx benchmark."""
        raise NotImplementedError

    def run_torch(self):
        """Run the benchmark using torch."""
        raise NotImplementedError

    def run_mlx(self):
        """Run the benchmark using mlx."""
        raise NotImplementedError

    def setup(
        self,
        framework: str,
        backend: str,
        dtype: str,
        compile: bool,
    ):
        """Setup the benchmark for the given framework and backend."""

        self._framework = framework
        self._backend = backend
        self._compile = compile

        if framework == "torch":
            self._device = torch.device(backend)

            self._dtype = dict(
                float32=torch.float32,
                bfloat16=torch.bfloat16,
                float16=torch.float16,
            )[dtype]

            torch.manual_seed(0)
            torch.set_default_device(self._device)
            torch.set_default_dtype(self._dtype)

            self.input_tensor = torch.rand(
                self.input_shape,
                device=self._device,
                dtype=self._dtype,
            )
            self.setup_torch()

        elif framework == "mlx":
            if backend == "metal":
                self._device = mx.Device(mx.DeviceType.gpu)
            elif backend == "cpu":
                self._device = mx.Device(mx.DeviceType.cpu)
            else:
                raise NotImplementedError(backend)

            self._dtype = dict(
                float32=mx.float32,
                bfloat16=mx.bfloat16,
                float16=mx.float16,
            )[dtype]

            mx.random.seed(0)
            mx.set_default_device(self._device)

            self.input_tensor = mx.random.normal(self.input_shape).astype(self._dtype)
            self.setup_mlx()

        else:
            raise NotImplementedError(f"Unknown framework {framework}")

    def run_once(self):
        """Run the benchmark once."""
        if self._framework == "torch":
            output: torch.Tensor = self.run_torch()

            if self._backend == "mps":
                torch.mps.synchronize()
            elif self._backend == "cuda":
                torch.cuda.synchronize()

        elif self._framework == "mlx":
            output: mx.array = self.run_mlx()
            mx.eval(output)

        elif self._framework is None:
            raise ValueError("Framework not set. Call setup() first!")
        else:
            raise NotImplementedError(f"Unknown framework {self._framework}")

    def teardown(self):
        """Teardown the benchmark."""
        del self.input_tensor

        if self._framework == "torch":
            del self.torch_function
            if self._backend == "mps":
                torch.mps.empty_cache()
            elif self._backend == "cuda":
                torch.cuda.empty_cache()

        if self._framework == "mlx":
            del self.mlx_function
            mx.clear_cache()

        # reset placeholders
        self.input_tensor = None
        self.torch_function = None
        self.mlx_function = None

        self._framework = None
        self._backend = None
        self._dtype = None
        self._device = None
        self._compile = None

        gc.collect()
