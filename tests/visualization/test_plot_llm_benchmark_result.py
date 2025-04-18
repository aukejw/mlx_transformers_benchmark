import itertools

import pandas as pd
import plotly.graph_objects as go
import pytest

from mtb.visualization.plot_llm_benchmark_result import show_llm_benchmark_data


@pytest.fixture
def sample_benchmark_data() -> pd.DataFrame:
    frameworks = ["torch_2.6.0_mps", "mlx_0.24.1_metal"]
    dtypes = ["float32", "float16", "bfloat16"]
    batch_sizes = [1, 8]
    num_prompt_tokens = [64, 128, 256]

    data = []
    for framework, dtype, batch_size, num_prompt_tokens in itertools.product(
        frameworks,
        dtypes,
        batch_sizes,
        num_prompt_tokens,
    ):
        generation_tps = batch_size
        prompt_time_sec = num_prompt_tokens * 0.1

        data.append(
            {
                "framework_backend": framework,
                "dtype": dtype,
                "batch_size": batch_size,
                "num_prompt_tokens": num_prompt_tokens,
                "generation_tps": generation_tps,
                "prompt_time_sec": prompt_time_sec,
            }
        )

    return pd.DataFrame(data)


@pytest.mark.parametrize("do_average_measurements", [True, False])
def test_show_benchmark_data_returns_figure(
    sample_benchmark_data, do_average_measurements
):
    fig = show_llm_benchmark_data(
        title="Title",
        measurements=sample_benchmark_data,
        do_average_measurements=do_average_measurements,
    )
    assert isinstance(fig, go.Figure)


def test_show_benchmark_data_with_specific_params(sample_benchmark_data):
    custom_dtypes = ("float32", "float16")
    custom_batch_sizes = (1,)

    fig = show_llm_benchmark_data(
        title="Title",
        measurements=sample_benchmark_data,
        dtypes=custom_dtypes,
        batch_sizes=custom_batch_sizes,
    )

    assert isinstance(fig, go.Figure)

    # Check that the figure has the correct number of subplots
    assert (
        len(fig.layout.annotations) == len(custom_dtypes) * len(custom_batch_sizes) * 3
    )

    # Check that all traces have the correct hovertemplate
    for trace in fig.data:
        if hasattr(trace, "hovertemplate"):
            assert "Batch size" in trace.hovertemplate
            assert "Num prompt tokens" in trace.hovertemplate
            assert "Prompt time" in trace.hovertemplate
            assert "Gen.speed" in trace.hovertemplate
