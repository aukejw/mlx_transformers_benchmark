import torch
import time
import mlx.core as mx


class BaseBenchmark:
    """Benchmark class containing entrypoints."""

    def __init__(self, name: str, input_shapes: list):
        self.name = name

        # placeholders for functions to call
        self.torch_function = None
        self.mlx_function = None
        self.compiled_mlx_function = None

        self.input_shapes = input_shapes

    def run_once(self, framework: str, device: str):
        """Run the benchmark once."""

        start_time = time.perf_counter()

        if framework == "torch":
            self.run_torch(
                device=device,
            )

        elif framework == "mlx":
            self.run_mlx(
                device=device,
                compile=True,
            )

        duration = time.perf_counter() - start_time
        return duration

    @torch.inference_mode()
    def run_torch(self, device: str):
        """Run the benchmark using torch."""

        self._run_torch(device)

        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
    
    def run_mlx(self, device: str, compile: bool):
        """Run the benchmark using mlx."""
        self._run_mlx(compile=compile)
        mx.eval()

    def _run_torch(self, device: torch.device):
        """Implement the torch benchmark."""
        raise NotImplementedError

    def _run_mlx(self, compile: bool):
        """Implement the mlx benchmark. Should end with `mx.eval`."""

        if compile and self.compiled_mlx_function
        raise NotImplementedError
