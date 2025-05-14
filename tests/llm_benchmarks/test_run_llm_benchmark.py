from typing import Dict, List
from unittest.mock import Mock

import pandas as pd
import pytest

from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark
from mtb.llm_benchmarks.models.base import ModelSpec
from mtb.llm_benchmarks.run_llm_benchmark import (
    run_benchmark,
    run_benchmark_for_framework,
)
from mtb.measurement import LlmBenchmarkMeasurement

MockModelSpec = ModelSpec(
    name="mock_model",
    num_params=1e9,
    prompt_formatter=lambda x: x,
    model_ids={
        "torch": {"float16": "mock_model_id"},
        "mlx": {"float16": "mock_model_id"},
        "lmstudio": {"float16": "mock_model_id"},
    },
)


class MockBenchmark(BaseLLMBenchmark):
    framework = "torch"

    def __init__(self, name: str):
        super().__init__(
            name=name,
            model_id="mock_model_id",
            backend="cpu",
            dtype="float16",
            prompt_formatter=lambda x: x,
            max_num_tokens=10,
        )

    def setup(self):
        pass

    def format_prompt(self, prompt: str):
        return [[0 for _ in range(len(prompt))]]

    def run_once(self, prompt: str) -> LlmBenchmarkMeasurement:
        return LlmBenchmarkMeasurement(
            response="response",
            num_prompt_tokens=1,
            num_generated_tokens=2,
            generation_tps=3,
            prompt_tps=4,
            prompt_time_sec=5,
            generation_time_sec=6,
            peak_memory_gib=7,
        )

    def teardown(self):
        pass


@pytest.fixture
def benchmark():
    return MockBenchmark(name=MockModelSpec.name)


def test_run_benchmark_for_framework(benchmark):
    measurements = run_benchmark_for_framework(
        benchmark=benchmark,
        batch_sizes=(1,),
        prompt_lengths=(64,),
        num_warmup_iterations=1,
        num_iterations=2,
        cooldown_time_fraction=0.1,
    )
    assert isinstance(measurements, List)
    assert isinstance(measurements[0], Dict)

    for key in [
        "generation_tps",
        "prompt_tps",
        "num_prompt_tokens",
        "num_generated_tokens",
        "prompt_time_sec",
        "generation_time_sec",
        "peak_memory_gib",
    ]:
        assert key in measurements[0]


def test_run_benchmark(monkeypatch, benchmark, tmp_path):
    # Mock create_benchmark to return the MockBenchmark instance
    mock_create_benchmark = Mock(return_value=benchmark)
    monkeypatch.setattr(
        "mtb.llm_benchmarks.run_llm_benchmark.create_benchmark",
        mock_create_benchmark,
    )

    output_path = tmp_path / "benchmark_results.csv"
    measurements_df = run_benchmark(
        model_spec=MockModelSpec,
        output_path=output_path,
        batch_sizes=(1,),
        prompt_lengths=(64,),
        framework="torch",
        backend="cpu",
        dtype="float16",
        num_warmup_iterations=1,
        num_iterations=1,
        max_num_tokens=10,
        cooldown_time_fraction=0.01,
    )
    assert output_path.exists()

    # we should have one measurement for torch, nothing else
    measurements_df = pd.read_csv(output_path)
    assert isinstance(measurements_df, pd.DataFrame)
    assert len(measurements_df) == 1
    assert measurements_df["framework"].iloc[0] == "torch"
    assert measurements_df["backend"].iloc[0] == "cpu"
