from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp


def show_llm_benchmark_data(
    title: str,
    measurements: pd.DataFrame,
    dtypes: List[str] = ("float32", "float16", "bfloat16"),
    batch_sizes: List[int] = (1,),
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
    y_metrics = {
        "prompt_time_sec": "Prompt time (s)",
        "generation_tps": "Gen. speed (tokens/s)",
        "current_memory_gb": "Memory (GB)",
    }

    fig = sp.make_subplots(
        rows=len(dtypes),
        cols=len(batch_sizes) * 3,
        subplot_titles=[
            f"{title} B={batch_size}"
            for dtype in dtypes
            for title in y_metrics.values()
            for batch_size in batch_sizes
        ],
        horizontal_spacing=0.05,
        vertical_spacing=0.075,
    )

    for row, dtype in enumerate(dtypes):
        for col, batch_size in enumerate(batch_sizes):
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
                        "num_prompt_tokens",
                        "prompt_time_sec",
                        "generation_tps",
                        "current_memory_gb",
                    ]
                ]
                filtered_data = (
                    filtered_data.groupby(
                        ["framework_backend", "batch_size", "num_prompt_tokens"],
                        observed=True,
                    )
                    .mean()
                    .reset_index()
                )

            # Show
            if not filtered_data.empty:
                for col_offset, y_metric_name in enumerate(y_metrics):
                    scatter = px.scatter(
                        filtered_data,
                        x="num_prompt_tokens",
                        y=y_metric_name,
                        color="framework_backend",
                        symbol="framework_backend",
                        custom_data=["batch_size"] + list(y_metrics.keys()),
                        title=f"dtype: {dtype}, batch_size: {batch_size}",
                    )

                    row_index = row + 1
                    column_index = col * len(y_metrics) + col_offset + 1

                    for trace in scatter["data"]:
                        fig.add_trace(trace, row=row_index, col=column_index)
                        fig.update_yaxes(
                            row=row_index,
                            col=column_index,
                            title_text=y_metrics[y_metric_name],
                        )

    # Update x and y axes layouts for all subplots
    fig.update_xaxes(
        type="log",
        tickvals=[4096, 2048, 1028, 512, 256, 128, 64, 32, 16],
        ticktext=["4096", "2048", "1028", "512", "256", "128", "64", "32", "16"],
    )
    fig.update_xaxes(
        row=len(dtypes),
        title_text="Num prompt tokens",
    )
    fig.update_yaxes(
        type="log",
        tickformat=".2g",
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
            "<b>Batch size:</b>           %{customdata[0]:.0f}<br>"
            "<b>Num prompt tokens:</b>    %{x:.0f}<br>"
            "<b>Prompt time (s):</b>      %{customdata[1]:.4f}<br>"
            "<b>Gen.speed (tokens/s):</b> %{customdata[2]:.4f}<br>"
            "<b>Memory (GB):</b>          %{customdata[3]:.4f}<br>"
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
