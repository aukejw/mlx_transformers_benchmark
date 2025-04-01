import time

import mlx.core as mx
import torch


class BaseBenchmark:
    """Benchmark class containing entrypoint functions.

    Each benchmark should implement setup and run functions:

      1. `_setup_torch`
      2. `_setup_mlx`
      3. `_run_torch`
      4. `_run_mlx`:

    """

    def __init__(self, name: str, input_shapes: list):
        self.name = name

        # placeholders for functions
        self.torch_function = None
        self.mlx_function = None

        self.input_shapes = input_shapes
        self.input_tensors = None

    def setup(self, framework: str, dtype: str, backend: str, compile: bool):
        """Setup the benchmark for the given framework and backend."""

        if framework == "torch":
            torch.set_default_device(torch.device(backend))

            dtype = dict(
                float32=torch.float32,
                bfloat16=torch.bfloat16,
                float16=torch.float16,
            )[dtype]

            self.input_tensors = [
                torch.rand(input_shape, device=backend, dtype=dtype)
                for input_shape in self.input_shapes
            ]
            self._setup_torch(backend=backend, dtype=dtype)

        elif framework == "mlx":
            if backend == "metal":
                device = mx.Device(mx.DeviceType.gpu)
            elif backend == "cpu":
                device = mx.Device(mx.DeviceType.cpu)

            mx.set_default_device(device)

            dtype = dict(
                float32=mx.float32,
                bfloat16=mx.bfloat16,
                float16=mx.float16,
            )[dtype]

            self.input_tensors = [
                mx.random.normal(shape=input_shape).astype(dtype)
                for input_shape in self.input_shapes
            ]
            self._setup_mlx(backend=backend, dtype=dtype, compile=compile)

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

    @torch.inference_mode()
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

    def _run_torch(self):
        """Implement the torch benchmark."""
        raise NotImplementedError

    def _run_mlx(self):
        """Implement the mlx benchmark."""
        raise NotImplementedError

    def teardown(self, framework: str, backend: str):
        """Cleanup."""
        del self.input_tensors

        if framework == "torch":
            del self.torch_function

            if backend == "mps":
                torch.mps.empty_cache()
            if backend == "cuda":
                torch.cuda.empty_cache()

        if framework == "mlx":
            del self.mlx_function
            mx.clear_cache()
