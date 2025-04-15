import mlx.core as mx
import mlx.nn
import pytest
import torch

from mtb.layer_benchmarks.base_layer_benchmark import BaseLayerBenchmark

try:
    # Check if MLX compilation is available
    mx.eval(mx.compile(mlx.nn.Linear(1, 1))(mx.ones((1, 1))))
    MLX_COMPILATION_UNAVAILABLE = False
except RuntimeError:
    MLX_COMPILATION_UNAVAILABLE = True


class LayerBenchmarkTest:
    """Define common tests for layer benchmarks."""

    @pytest.fixture
    def benchmark(self) -> BaseLayerBenchmark:
        raise NotImplementedError

    def benchmark_setup_run_teardown(  # Renamed from setup_run_teardown
        self,
        benchmark: BaseLayerBenchmark,
        framework: str,
        backend: str,
        dtype: str = "float32",
        compile: bool = False,
    ):
        benchmark.setup(
            framework=framework,
            backend=backend,
            dtype=dtype,
            compile=compile,
        )
        benchmark.run_once()
        benchmark.teardown()

    def test_torch_cpu(self, benchmark):
        self.benchmark_setup_run_teardown(benchmark, "torch", "cpu")

    def test_mlx_cpu(self, benchmark):
        self.benchmark_setup_run_teardown(benchmark, "mlx", "cpu")

    @pytest.mark.skipif(
        MLX_COMPILATION_UNAVAILABLE, reason="MLX compilation unavailable"
    )
    def test_mlx_cpu_compiled(self, benchmark):
        self.benchmark_setup_run_teardown(benchmark, "mlx", "cpu", compile=True)

    @pytest.mark.skipif(not torch.mps.is_available(), reason="mps not available")
    def test_torch_mps(self, benchmark):
        self.benchmark_setup_run_teardown(benchmark, "torch", "mps")

    @pytest.mark.skipif(not torch.mps.is_available(), reason="gpu not available")
    def test_mlx_metal(self, benchmark):
        self.benchmark_setup_run_teardown(benchmark, "mlx", "metal")

    @pytest.mark.skipif(
        MLX_COMPILATION_UNAVAILABLE, reason="MLX compilation unavailable"
    )
    @pytest.mark.skipif(not torch.mps.is_available(), reason="gpu not available")
    def test_mlx_metal_compiled(self, benchmark):
        self.benchmark_setup_run_teardown(benchmark, "mlx", "metal", compile=True)
