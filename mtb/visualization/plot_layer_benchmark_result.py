from typing import Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from natsort import natsorted

from mtb.visualization.symbol_and_color import add_category_to_colormap


def show_layer_benchmark_data(
    title: str,
    measurements: pd.DataFrame,
    dtypes: Tuple[str] = ("float32", "float16", "bfloat16"),
    batch_sizes: Tuple[int] = (1, 8, 16, 32, 64),
    do_average_measurements: bool = True,
) -> go.Figure:
    """Visualize benchmark data in a single page.

    Args:
        title: Title of the benchmark task.
        measurements: DataFrame containing benchmark measurements.
        dtypes: Tuple of data types to show. One dtype = one row.
        batch_sizes: Tuple of batchsizes to show. One batchsize = one column.
        do_average_measurements: If False, show all individual measurements.

    Returns:
        The created figure.

    """
    frameworks = natsorted(measurements["framework_backend"].unique())
    for framework in frameworks:
        add_category_to_colormap(framework)

    fig = sp.make_subplots(
        rows=len(dtypes),
        cols=len(batch_sizes),
        subplot_titles=[
            f"{dtype}, batch_size={batch_size}"
            for dtype in dtypes
            for batch_size in batch_sizes
        ],
        horizontal_spacing=0.05,
        vertical_spacing=0.075,
    )

    for row, dtype in enumerate(dtypes, start=1):
        for col, batch_size in enumerate(batch_sizes, start=1):
            # Select data
            filtered_data = measurements[
                (measurements["dtype"] == dtype)
                & (measurements["batch_size"] == batch_size)
            ]
            if do_average_measurements:
                filtered_data = filtered_data[
                    [
                        "framework_backend",
                        "batch_size",
                        "sequence_length",
                        "duration_ms",
                    ]
                ]
                filtered_data = (
                    filtered_data.groupby(
                        ["framework_backend", "batch_size", "sequence_length"],
                        observed=True,
                    )
                    .mean()
                    .reset_index()
                )

            # Show
            if not filtered_data.empty:
                scatter = px.scatter(
                    filtered_data,
                    x="sequence_length",
                    y="duration_ms",
                    color="framework_backend",
                    symbol="framework_backend",
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

    # Optimize legend entries, layout
    legend_entries = set()
    for trace in fig.data:
        if trace.name not in legend_entries:
            legend_entries.add(trace.name)
        else:
            trace.showlegend = False

    fig.update_layout(
        height=800,
        width=1600,
        title_text=f"Benchmark {title}",
        title=dict(
            y=0.98,
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(size=18),
        ),
        margin=dict(
            t=80,
            l=50,
            r=50,
            b=60,
        ),
        showlegend=True,
        legend=dict(
            x=1.05,
            y=1,
            font=dict(size=14),
            tracegroupgap=5,
        ),
        font=dict(size=10),
    )

    # Reduce subplot title font size
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=14)

    # Add a hover template, already shows framework_backend by default
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
