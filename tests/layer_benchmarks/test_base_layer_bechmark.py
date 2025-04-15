from unittest.mock import patch

import mlx.nn
import pytest
import torch.nn

from mtb.layer_benchmarks.base_layer_benchmark import BaseLayerBenchmark


class MockBenchmark(BaseLayerBenchmark):
    def run_torch(self):
        return self.input_tensor

    def run_mlx(self):
        return self.input_tensor


@pytest.fixture
def benchmark():
    return MockBenchmark(name="mock", input_shape=(1, 3, 5))


def test_base_benchmark_illegal_inputs(benchmark):
    with pytest.raises(NotImplementedError):
        benchmark.setup(
            framework="illegal",
            dtype="float32",
            backend="metal",
            compile=True,
        )

    with pytest.raises(NotImplementedError):
        benchmark.setup(
            framework="mlx",
            dtype="float32",
            backend="illegal",
            compile=True,
        )

    with pytest.raises(NotImplementedError):
        benchmark._framework = "illegal"
        benchmark.run_once()

    with pytest.raises(ValueError):
        benchmark._framework = None
        benchmark.run_once()


def test_syncs_are_called(benchmark: MockBenchmark):
    with patch("torch.mps.synchronize") as mock_mps_sync:
        benchmark._framework = "torch"
        benchmark._backend = "mps"
        benchmark.run_once()
        mock_mps_sync.assert_called_once()

    with patch("torch.cuda.synchronize") as mock_cuda_sync:
        benchmark._framework = "torch"
        benchmark._backend = "cuda"
        benchmark.run_once()
        mock_cuda_sync.assert_called_once()

    with patch("mlx.core.eval") as mock_mx_eval:
        benchmark._framework = "mlx"
        benchmark._backend = "metal"
        benchmark.run_once()
        mock_mx_eval.assert_called_once()


def test_teardown(benchmark: MockBenchmark):
    with patch("torch.mps.empty_cache") as mock_mps_empty_cache:
        benchmark._framework = "torch"
        benchmark._backend = "mps"
        benchmark.torch_function = torch.nn.Identity()
        benchmark.teardown()
        mock_mps_empty_cache.assert_called_once()
        assert benchmark.torch_function is None

    with patch("torch.cuda.empty_cache") as mock_cuda_empty_cache:
        benchmark._framework = "torch"
        benchmark._backend = "cuda"
        benchmark.torch_function = torch.nn.Identity()
        benchmark.teardown()
        mock_cuda_empty_cache.assert_called_once()
        assert benchmark.torch_function is None

    with patch("mlx.core.clear_cache") as mock_mx_clear_cache:
        benchmark._framework = "mlx"
        benchmark._backend = "metal"
        benchmark.mlx_function = mlx.nn.Identity()
        benchmark.teardown()
        mock_mx_clear_cache.assert_called_once()
        assert benchmark.mlx_function is None
