import time
from unittest.mock import Mock

import mlx.nn
import pandas as pd
import pytest
import torch.nn

from mtb.layer_benchmarks.base_layer_benchmark import BaseLayerBenchmark
from mtb.layer_benchmarks.run_layer_benchmark import (
    run_benchmark,
    run_benchmark_for_framework,
)


class MockBenchmark(BaseLayerBenchmark):
    def setup_torch(self):
        self.torch_function = torch.nn.Identity()

    def setup_mlx(self):
        self.mlx_function = mlx.nn.Identity()

    def run_torch(self):
        return self.torch_function(self.input_tensor)

    def run_mlx(self):
        return self.mlx_function(self.input_tensor)


@pytest.fixture
def benchmark():
    return MockBenchmark(name="mock", feature_dim=16)


def test_run_benchmark_for_framework(benchmark):
    measurements = run_benchmark_for_framework(
        benchmark=benchmark,
        batch_sizes=(1,),
        sequence_lengths=(3,),
        framework="torch",
        backend="cpu",
        dtype="float16",
        num_warmup_iterations=1,
        num_iterations=2,
        compile=False,
    )
    assert isinstance(measurements, list) and len(measurements) == 1

    measurement = measurements[0]
    assert measurement["batch_size"] == 1
    assert measurement["sequence_length"] == 3
    assert measurement["duration_ms"] > 0


def test_run_benchmark_for_framework_slowiterations(benchmark):
    def slow_run_torch():
        time.sleep(0.2)

    benchmark.run_torch = slow_run_torch

    measurements = run_benchmark_for_framework(
        benchmark=benchmark,
        batch_sizes=(1,),
        sequence_lengths=(3,),
        min_runtime_ms=0.1,
        framework="torch",
        backend="cpu",
        dtype="float16",
        num_warmup_iterations=1,
        num_iterations=2,
        compile=False,
    )
    assert isinstance(measurements, list)


def test_run_benchmark_on_cpu(benchmark):
    measurements_df = run_benchmark(
        benchmark=benchmark,
        batch_sizes=(1,),
        sequence_lengths=(3,),
        dtype="float16",
        num_warmup_iterations=1,
        num_iterations=2,
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
        batch_sizes=(1,),
        sequence_lengths=(3,),
        dtype="bfloat16",
        num_warmup_iterations=1,
        num_iterations=2,
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
        batch_sizes=(1,),
        sequence_lengths=(3,),
        dtype="bfloat16",
        num_warmup_iterations=1,
        num_iterations=2,
        run_torch_cuda=True,
    )
    assert isinstance(measurements_df, pd.DataFrame)
    assert len(measurements_df) == 1


def test_run_benchmark_calls_with_correct_args(monkeypatch, benchmark):
    mock_measurements = [
        dict(
            duration_ms=1,
        )
    ]

    # Setup the monkeypatch to replace run_benchmark_for_framework
    mock_run_benchmark_for_framework = Mock(
        return_value=mock_measurements,
    )
    monkeypatch.setattr(
        "mtb.layer_benchmarks.run_layer_benchmark.run_benchmark_for_framework",
        mock_run_benchmark_for_framework,
    )

    # Test parameters
    num_warmup_iterations = 1
    num_iterations = 1
    min_runtime_ms = 500
    dtype = "float16"

    # Define all backend options and their expected arguments
    backend_options = [
        {
            "option": "run_torch_cpu",
            "framework": "torch",
            "backend": "cpu",
            "compile": False,
        },
        {
            "option": "run_torch_mps",
            "framework": "torch",
            "backend": "mps",
            "compile": False,
        },
        {
            "option": "run_torch_cuda",
            "framework": "torch",
            "backend": "cuda",
            "compile": False,
        },
        {
            "option": "run_mlx_cpu",
            "framework": "mlx",
            "backend": "cpu",
            "compile": False,
        },
        {
            "option": "run_mlx_metal",
            "framework": "mlx",
            "backend": "metal",
            "compile": False,
        },
        {
            "option": "run_mlx_metal_compiled",
            "framework": "mlx",
            "backend": "metal",
            "compile": True,
        },
    ]

    for backend in backend_options:
        mock_run_benchmark_for_framework.reset_mock()

        kwargs = {
            "benchmark": benchmark,
            "batch_sizes": (1,),
            "sequence_lengths": (3,),
            "num_warmup_iterations": num_warmup_iterations,
            "num_iterations": num_iterations,
            "dtype": dtype,
            "cooldown_time_fraction": 0.1,
        }
        kwargs[backend["option"]] = True

        run_benchmark(**kwargs)

        mock_run_benchmark_for_framework.assert_called_once_with(
            benchmark=benchmark,
            batch_sizes=(1,),
            sequence_lengths=(3,),
            cooldown_time_fraction=0.1,
            num_warmup_iterations=num_warmup_iterations,
            num_iterations=num_iterations,
            min_runtime_ms=min_runtime_ms,
            dtype=dtype,
            framework=backend["framework"],
            backend=backend["backend"],
            compile=backend["compile"],
        )

    # Test with multiple options enabled
    mock_run_benchmark_for_framework.reset_mock()
