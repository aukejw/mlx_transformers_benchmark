from typing import Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp


def show_benchmark_data(
    title: str,
    measurements: pd.DataFrame,
    dtypes: Tuple[str] = ("float32", "float16", "bfloat16"),
    batch_sizes: Tuple[int] = (1, 8, 16, 32, 64),
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
                    y="mean_ms",
                    color="framework_backend",
                    custom_data=["batch_size"],
                    title=f"dtype: {dtype}, batch_size: {batch_size}",
                )

                for trace in scatter["data"]:
                    fig.add_trace(trace, row=row, col=col)

    # Update x and y axes layouts for all subplots
    fig.update_xaxes(
        type="log",
        tickvals=[512, 256, 128, 64],
        ticktext=["512", "256", "128", "64"],
    )
    fig.update_xaxes(
        row=len(dtypes),
        title_text="Sequence length (tokens)",
    )
    fig.update_yaxes(
        type="log",
        tickformat=".2g",
    )
    fig.update_yaxes(
        col=1,
        title_text="Runtime (ms)",
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
        font=dict(size=10),
    )

    # Add a hover template
    fig.update_traces(
        hovertemplate=(
            "<b>Batch size:</b>  %{customdata[0]:.0f}<br>"
            "<b>Seq. length:</b> %{x:.0f}<br>"
            "<b>Runtime:</b>     %{y:.4f} ms"
        ),
        mode="markers",
    )
    fig.update_layout(
        hoverlabel=dict(
            font_family="Menlo, monospace",
            font_size=14,
        )
    )
    return fig
