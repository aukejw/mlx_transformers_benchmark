import itertools

import pandas as pd
import plotly.graph_objects as go
import pytest

from mtb.visualization.plot_benchmark_result import show_benchmark_data


@pytest.fixture
def sample_benchmark_data() -> pd.DataFrame:
    frameworks = ["torch_2.6.0_mps", "mlx_0.24.1_metal"]
    dtypes = ["float32", "float16", "bfloat16"]
    batch_sizes = [1, 8]
    sequence_lengths = [64]

    data = []
    for framework, dtype, batch_size, seq_len in itertools.product(
        frameworks, dtypes, batch_sizes, sequence_lengths
    ):
        mean_ms = (seq_len * batch_size) / 1e3

        data.append(
            {
                "framework_backend": framework,
                "dtype": dtype,
                "batch_size": batch_size,
                "sequence_length": seq_len,
                "mean_ms": mean_ms,
            }
        )

    return pd.DataFrame(data)


@pytest.mark.parametrize("do_average_measurements", [True, False])
def test_show_benchmark_data_returns_figure(
    sample_benchmark_data, do_average_measurements
):
    fig = show_benchmark_data(
        title="Title",
        measurements=sample_benchmark_data,
        do_average_measurements=do_average_measurements,
    )
    assert isinstance(fig, go.Figure)


def test_show_benchmark_data_with_specific_params(sample_benchmark_data):
    custom_dtypes = ("float32", "float16")
    custom_batch_sizes = (1, 16, 64)

    fig = show_benchmark_data(
        title="Title",
        measurements=sample_benchmark_data,
        dtypes=custom_dtypes,
        batch_sizes=custom_batch_sizes,
    )

    assert isinstance(fig, go.Figure)

    # Check that the figure has the correct number of subplots
    assert len(fig.layout.annotations) == len(custom_dtypes) * len(custom_batch_sizes)

    # Check that all traces have the correct hovertemplate
    for trace in fig.data:
        if hasattr(trace, "hovertemplate"):
            assert "Batch size" in trace.hovertemplate
            assert "Seq. length" in trace.hovertemplate
            assert "Runtime" in trace.hovertemplate
