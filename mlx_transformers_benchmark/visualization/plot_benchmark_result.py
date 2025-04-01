from typing import Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp


def show_benchmark_data(
    title: str,
    measurements: pd.DataFrame,
    dtypes: Tuple[str] = ("float32", "float16", "bfloat16"),
    batch_sizes: Tuple[int] = (1, 16, 32, 64, 128),
) -> go.Figure:
    """Visualize benchmark data in a single page.

    Args:
        title: Title of the benchmark task.
        measurements: DataFrame containing benchmark measurements.
        dtypes: Tuple of data types to show. One dtype = one row.
        batch_sizes: Tuple of batchsizes to show. One batchsize = one column.

    Returns:
        The created figure.

    """

    fig = sp.make_subplots(
        rows=len(dtypes),
        cols=len(batch_sizes),
        subplot_titles=[
            f"{dtype}, batch_size={batch_size}"
            for dtype in dtypes
            for batch_size in batch_sizes
        ],
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )

    for row, dtype in enumerate(dtypes, start=1):
        for col, batch_size in enumerate(batch_sizes, start=1):
            filtered_data = measurements[
                (measurements["dtype"] == dtype)
                & (measurements["batch_size"] == batch_size)
            ]

            if not filtered_data.empty:
                scatter = px.scatter(
                    filtered_data,
                    x="sequence_length",
                    y="median_ms",
                    color="framework_backend",
                    title=f"dtype: {dtype}, batch_size: {batch_size}",
                    labels={
                        "name": "Benchmark",
                        "framework_backend": "Compute",
                        "batch_size": "Batch size",
                        "sequence_length": "Sequence length",
                        "mean_ms": "Mean runtime (ms)",
                    },
                )

                for trace in scatter["data"]:
                    fig.add_trace(trace, row=row, col=col)

            # Update x-axis for the current subplot
            fig.update_xaxes(
                tickvals=[512, 256, 128, 64],
                ticktext=["512", "256", "128", "64"],
                row=row,
                col=col,
            )

    # optimize legend entries
    legend_entries = set()
    for trace in fig.data:
        if trace.name not in legend_entries:
            legend_entries.add(trace.name)
        else:
            trace.showlegend = False

    fig.update_layout(
        height=900,
        width=1500,
        title_text=f"Benchmark '{title}'",
        showlegend=True,
        legend=dict(
            x=1.05,
            y=1,
            font=dict(size=14),
            tracegroupgap=5,
        ),
        font=dict(size=12),
    )
    return fig
