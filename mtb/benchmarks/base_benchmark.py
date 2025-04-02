from typing import Tuple

import mlx.core as mx
import torch


class BaseBenchmark:
    """Benchmark class containing entrypoint functions.

    Each benchmark should implement setup and run four functions:

      1. `_setup_torch`
      2. `_setup_mlx`
      3. `_run_torch`
      4. `_run_mlx`:

    """

    def __init__(
        self,
        name: str,
        input_shape: Tuple[int, int, int],
    ):
        self.name = name

        self._batch_size = input_shape[0]
        self._num_tokens = input_shape[1]
        self._num_features = input_shape[2]

        # placeholders for functions
        self.torch_function = None
        self.mlx_function = None

        self.input_shape = input_shape
        self.input_tensor = None

    def setup(
        self,
        framework: str,
        dtype: str,
        backend: str,
        compile: bool,
    ):
        """Setup the benchmark for the given framework and backend."""

        if framework == "torch":
            torch.manual_seed(0)
            torch.set_default_device(torch.device(backend))

            dtype = dict(
                float32=torch.float32,
                bfloat16=torch.bfloat16,
                float16=torch.float16,
            )[dtype]

            self.input_tensor = torch.rand(
                self.input_shape, device=backend, dtype=dtype
            )
            self._setup_torch(backend=backend, dtype=dtype)

        elif framework == "mlx":
            if backend == "metal":
                device = mx.Device(mx.DeviceType.gpu)
            elif backend == "cpu":
                device = mx.Device(mx.DeviceType.cpu)
            else:
                raise NotImplementedError(backend)

            mx.random.seed(0)
            mx.set_default_device(device)

            dtype = dict(
                float32=mx.float32,
                bfloat16=mx.bfloat16,
                float16=mx.float16,
            )[dtype]

            self.input_tensor = mx.random.normal(self.input_shape).astype(dtype)
            self._setup_mlx(backend=backend, dtype=dtype, compile=compile)

        else:
            raise NotImplementedError(framework)

    def _setup_torch(self, backend: str, dtype: str):
        """Setup the torch benchmark."""
        raise NotImplementedError

    def _setup_mlx(self, backend: str, dtype: str, compile: bool):
        """Setup the mlx benchmark."""
        raise NotImplementedError

    def run_once(self, framework: str, backend: str):
        """Run the benchmark once."""
        if framework == "torch":
            self.run_torch(backend)
        elif framework == "mlx":
            self.run_mlx(backend)
        else:
            raise NotImplementedError

    def run_torch(self, backend: str):
        """Run the benchmark using torch."""

        output: torch.Tensor = self._run_torch(backend)

        if backend == "mps":
            torch.mps.synchronize()
        elif backend == "cuda":
            torch.cuda.synchronize()

    def run_mlx(self, backend: str):
        """Run the benchmark using mlx."""

        output: mx.array = self._run_mlx(backend)
        mx.eval(output)

    def _run_torch(self, backend: str):
        """Implement the torch benchmark."""
        raise NotImplementedError

    def _run_mlx(self, backend: str):
        """Implement the mlx benchmark."""
        raise NotImplementedError

    def teardown(self, framework: str, backend: str):
        """Cleanup."""
        del self.input_tensor

        if framework == "torch":
            del self.torch_function

            if backend == "mps":
                torch.mps.empty_cache()
            if backend == "cuda":
                torch.cuda.empty_cache()

        if framework == "mlx":
            del self.mlx_function

            mx.clear_cache()
