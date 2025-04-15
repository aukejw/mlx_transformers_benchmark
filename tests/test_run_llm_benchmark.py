import time
from typing import Dict, List
from unittest.mock import Mock

import pandas as pd
import pytest
from transformers import BatchEncoding

from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark
from mtb.run_llm_benchmark import run_benchmark, run_benchmark_for_framework


class MockBenchmark(BaseLLMBenchmark):
    def setup_torch(self):
        pass

    def setup_mlx(self):
        pass

    def set_prompt(self, prompt: str, batch_size: int) -> BatchEncoding:
        self.model_input = BatchEncoding(
            data={
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]],
            },
            tensor_type="pt",
        )

    def run_torch_generate(self):
        time.sleep(0.01)
        return dict(
            generation_tps=1,
            prompt_tps=2,
            peak_memory_gb=3,
        )

    def run_mlx_generate(self):
        time.sleep(0.01)
        return dict(
            generation_tps=4,
            prompt_tps=5,
            peak_memory_gb=6,
        )


@pytest.fixture
def benchmark():
    return MockBenchmark(
        name="mock",
        max_num_tokens=10,
    )


def test_run_benchmark_for_framework(benchmark):
    measurements = run_benchmark_for_framework(
        benchmark=benchmark,
        batch_sizes=(1,),
        prompts=["prompt"],
        cooldown_time_fraction=0.1,
        framework="torch",
        backend="cpu",
        dtype="float16",
        num_warmup_iterations=1,
        num_iterations=2,
    )
    assert isinstance(measurements, List)
    assert isinstance(measurements[0], Dict)

    for key in [
        "generation_tps",
        "prompt_tps",
        "peak_memory_gb",
        "num_prompt_tokens",
    ]:
        assert key in measurements[0]


def test_run_benchmark(benchmark):
    measurements_df = run_benchmark(
        benchmark=benchmark,
        batch_sizes=(1,),
        prompts=["prompt"],
        cooldown_time_fraction=0.1,
        dtype="float16",
        num_warmup_iterations=1,
        num_iterations=1,
        run_torch_cpu=True,
        run_mlx_cpu=True,
    )
    assert isinstance(measurements_df, pd.DataFrame)

    # one for torch, one for mlx
    assert len(measurements_df) == 2


def test_run_benchmark_calls_with_correct_args(monkeypatch, benchmark):
    mock_measurements = [
        dict(
            generation_tps=1,
            prompt_tps=2,
            peak_memory_gb=3,
            num_prompt_tokens=4,
        )
    ]

    # Setup the monkeypatch to replace run_benchmark_for_framework
    mock_run_benchmark_for_framework = Mock(
        return_value=mock_measurements,
    )
    monkeypatch.setattr(
        "mtb.run_llm_benchmark.run_benchmark_for_framework",
        mock_run_benchmark_for_framework,
    )

    # Test parameters
    num_warmup_iterations = 1
    num_iterations = 1
    dtype = "float16"

    # Define all backend options and their expected arguments
    backend_options = [
        {
            "option": "run_torch_cpu",
            "framework": "torch",
            "backend": "cpu",
        },
        {
            "option": "run_torch_mps",
            "framework": "torch",
            "backend": "mps",
        },
        {
            "option": "run_torch_cuda",
            "framework": "torch",
            "backend": "cuda",
        },
        {
            "option": "run_mlx_cpu",
            "framework": "mlx",
            "backend": "cpu",
        },
        {
            "option": "run_mlx_metal",
            "framework": "mlx",
            "backend": "metal",
        },
    ]

    for backend in backend_options:
        mock_run_benchmark_for_framework.reset_mock()

        kwargs = {
            "benchmark": benchmark,
            "batch_sizes": (1,),
            "prompts": ["prompt"],
            "cooldown_time_fraction": 0.1,
            "num_warmup_iterations": num_warmup_iterations,
            "num_iterations": num_iterations,
            "dtype": dtype,
        }
        kwargs[backend["option"]] = True

        run_benchmark(**kwargs)

        mock_run_benchmark_for_framework.assert_called_once_with(
            benchmark=benchmark,
            batch_sizes=(1,),
            prompts=["prompt"],
            cooldown_time_fraction=0.1,
            num_warmup_iterations=num_warmup_iterations,
            num_iterations=num_iterations,
            dtype=dtype,
            framework=backend["framework"],
            backend=backend["backend"],
        )

    # Test with multiple options enabled
    mock_run_benchmark_for_framework.reset_mock()
