import pytest

from mtb.benchmarks.base_benchmark import BaseBenchmark


@pytest.fixture
def benchmark():
    return BaseBenchmark(name="mock", input_shape=(1, 3, 5))


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
        benchmark.run_once(framework="illegal", backend="metal")
