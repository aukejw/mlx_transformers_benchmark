import mlx.nn
import pandas as pd
import pytest
import torch.nn

from mtb.benchmarks.base_benchmark import BaseBenchmark
from mtb.measurement import Measurement
from mtb.run_benchmark import run_benchmark, run_benchmark_for_framework


class MockBenchmark(BaseBenchmark):
    def _setup_torch(self, backend: str, dtype: str):
        self.torch_function = torch.nn.Identity()

    def _setup_mlx(self, backend: str, dtype: str, compile: bool):
        self.mlx_function = mlx.nn.Identity()

    def _run_torch(self, backend: str):
        return self.torch_function(self.input_tensor)

    def _run_mlx(self, backend: str):
        return self.mlx_function(self.input_tensor)


@pytest.fixture
def benchmark():
    return MockBenchmark(name="mock", input_shape=(1, 3, 16))


def test_run_benchmark_for_framework(benchmark):
    measurement = run_benchmark_for_framework(
        benchmark=benchmark,
        framework="torch",
        backend="cpu",
        dtype="float16",
        num_warmup_iterations=1,
        num_iterations=2,
        num_repeats=2,
        compile=False,
    )
    assert isinstance(measurement, Measurement)


def test_run_benchmark_on_cpu(benchmark):
    measurements_df = run_benchmark(
        benchmark=benchmark,
        dtype="float16",
        num_warmup_iterations=1,
        num_iterations=2,
        num_repeats=2,
        run_torch_cpu=True,
        run_mlx_cpu=True,
    )
    assert isinstance(measurements_df, pd.DataFrame)

    # one for torch, one for mlx
    assert len(measurements_df) == 2


@pytest.mark.skipif(not torch.mps.is_available(), reason="Metal is not available")
def test_run_benchmark_on_metal(benchmark):
    measurements_df = run_benchmark(
        benchmark=benchmark,
        dtype="bfloat16",
        num_warmup_iterations=1,
        num_iterations=2,
        num_repeats=2,
        run_torch_mps=True,
        run_mlx_metal=True,
        run_mlx_metal_compiled=True,
    )
    assert isinstance(measurements_df, pd.DataFrame)
    assert len(measurements_df) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_run_benchmark_on_cuda(benchmark):
    measurements_df = run_benchmark(
        benchmark=benchmark,
        dtype="bfloat16",
        num_warmup_iterations=1,
        num_iterations=2,
        num_repeats=2,
        run_torch_cuda=True,
    )
    assert isinstance(measurements_df, pd.DataFrame)
    assert len(measurements_df) == 1
