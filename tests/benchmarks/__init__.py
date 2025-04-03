import pytest
import torch

from mtb.benchmarks.base_benchmark import BaseBenchmark


class BenchmarkTest:
    """Define common tests for benchmarks."""

    @pytest.fixture
    def benchmark(self) -> BaseBenchmark:
        raise NotImplementedError

    def benchmark_setup_run_teardown(  # Renamed from setup_run_teardown
        self,
        benchmark: BaseBenchmark,
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

    def test_mlx_cpu_compiled(self, benchmark):
        self.benchmark_setup_run_teardown(benchmark, "mlx", "cpu", compile=True)

    @pytest.mark.skipif(not torch.mps.is_available(), reason="mps not available")
    def test_torch_mps(self, benchmark):
        self.benchmark_setup_run_teardown(benchmark, "torch", "mps")

    @pytest.mark.skipif(not torch.mps.is_available(), reason="gpu not available")
    def test_mlx_metal(self, benchmark):
        self.benchmark_setup_run_teardown(benchmark, "mlx", "metal")

    @pytest.mark.skipif(not torch.mps.is_available(), reason="gpu not available")
    def test_mlx_metal_compiled(self, benchmark):
        self.benchmark_setup_run_teardown(benchmark, "mlx", "metal", compile=True)
