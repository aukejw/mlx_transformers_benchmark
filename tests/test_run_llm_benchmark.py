from typing import Dict, List
from unittest.mock import Mock

import mlx.core as mx
import pandas as pd
import pytest
import torch
from transformers import BatchEncoding

from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark
from mtb.measurement import LlmBenchmarkMeasurement
from mtb.run_llm_benchmark import run_benchmark, run_benchmark_for_framework


class MockBenchmark(BaseLLMBenchmark):
    name = "mock"
    num_params = int(1e6)

    dtype_to_model_id = {
        torch.float16: "mock_model_id",
        mx.float16: "mock_model_id",
    }

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
        return LlmBenchmarkMeasurement(
            response="response",
            num_prompt_tokens=1,
            num_generated_tokens=2,
            generation_tps=3,
            prompt_tps=4,
            prompt_time_sec=5,
            generation_time_sec=6,
        )

    def run_mlx_generate(self):
        return LlmBenchmarkMeasurement(
            response="response",
            num_prompt_tokens=1,
            num_generated_tokens=2,
            generation_tps=3,
            prompt_tps=4,
            prompt_time_sec=5,
            generation_time_sec=6,
        )


@pytest.fixture
def benchmark():
    return MockBenchmark(max_num_tokens=10)


def test_run_benchmark_for_framework(benchmark):
    measurements = run_benchmark_for_framework(
        benchmark=benchmark,
        batch_sizes=(1,),
        dtype="float16",
        prompts=["prompt"],
        cooldown_time_fraction=0.1,
        framework="torch",
        backend="cpu",
        num_warmup_iterations=1,
        num_iterations=2,
    )
    assert isinstance(measurements, List)
    assert isinstance(measurements[0], Dict)

    for key in [
        "generation_tps",
        "prompt_tps",
        "num_prompt_tokens",
    ]:
        assert key in measurements[0]


def test_run_benchmark(benchmark, tmp_path):
    output_path = tmp_path / "benchmark_results.csv"
    measurements_df = run_benchmark(
        benchmark=benchmark,
        output_path=output_path,
        batch_sizes=(1,),
        prompts=["prompt"],
        cooldown_time_fraction=0.1,
        dtypes=("float16",),
        num_warmup_iterations=1,
        num_iterations=1,
        run_torch_cpu=True,
        run_mlx_cpu=True,
    )
    assert output_path.exists()

    # one for torch, one for mlx
    measurements_df = pd.read_csv(output_path)
    assert isinstance(measurements_df, pd.DataFrame)
    assert len(measurements_df) == 2


def test_run_benchmark_calls_with_correct_args(monkeypatch, benchmark, tmp_path):
    mock_measurements = [
        dict(
            generation_tps=1,
            prompt_tps=2,
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
    dtypes = ("float16",)

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

    output_path = tmp_path / "benchmark_results.csv"
    for backend in backend_options:
        mock_run_benchmark_for_framework.reset_mock()

        kwargs = {
            "benchmark": benchmark,
            "output_path": output_path,
            "batch_sizes": (1,),
            "dtypes": dtypes,
            "prompts": ["prompt"],
            "cooldown_time_fraction": 0.1,
            "num_warmup_iterations": num_warmup_iterations,
            "num_iterations": num_iterations,
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
            dtype=dtypes[0],
            framework=backend["framework"],
            backend=backend["backend"],
        )

    # Test with multiple options enabled
    mock_run_benchmark_for_framework.reset_mock()
