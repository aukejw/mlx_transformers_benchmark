from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from natsort import natsorted

from mtb.visualization.symbol_and_color import (
    add_category_to_colormap,
    get_symbol_and_color_map,
)


def show_llm_benchmark_data(
    title: str,
    measurements: pd.DataFrame,
    dtypes: List[str] = ("bfloat16",),
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
    frameworks = natsorted(measurements["framework_backend"].unique())
    for framework in frameworks:
        add_category_to_colormap(framework)

    metrics_of_interest = [
        "prompt_tps",
        "generation_tps",
        "prompt_time_sec",
        "generation_time_sec",
        "total_time_sec",
    ]
    y_metrics = {
        "prompt_time_sec": "Prompt time (s)",
        "total_time_sec": "Prompt + 100 tokens time (s)",
        "generation_tps": "Gen. speed (tokens/s)",
    }

    fig = sp.make_subplots(
        rows=len(dtypes),
        cols=len(batch_sizes) * len(y_metrics),
        subplot_titles=[
            f"{title}, {dtype}, B={batch_size}"
            for dtype in dtypes
            for title in y_metrics.values()
            for batch_size in batch_sizes
        ],
        horizontal_spacing=0.075,
        vertical_spacing=0.075,
    )

    color_map, symbol_map = get_symbol_and_color_map()

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
                    ]
                    + metrics_of_interest
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
                filtered_data = filtered_data.sort_values(
                    by=["framework_backend", "num_prompt_tokens", "batch_size"],
                )
                for col_offset, y_metric_name in enumerate(y_metrics):
                    row_index = row + 1
                    column_index = col * len(y_metrics) + col_offset + 1

                    scatter = px.scatter(
                        filtered_data,
                        x="num_prompt_tokens",
                        y=y_metric_name,
                        color="framework_backend",
                        symbol="framework_backend",
                        category_orders={"framework_backend": frameworks},
                        color_discrete_map=color_map,
                        symbol_map=symbol_map,
                        custom_data=["batch_size"] + metrics_of_interest,
                        title=f"dtype: {dtype}, batch_size: {batch_size}",
                    )
                    for trace in scatter["data"]:
                        fig.add_trace(trace, row=row_index, col=column_index)
                        fig.update_yaxes(
                            row=row_index,
                            col=column_index,
                            title_text=y_metrics[y_metric_name],
                        )
            else:
                # Disable the subplot, show text "no data available"
                for col_offset in range(len(y_metrics)):
                    row_index = row + 1
                    column_index = col * len(y_metrics) + col_offset + 1

                    # disable the subplot
                    fig.update_xaxes(
                        row=row_index,
                        col=column_index,
                        visible=False,
                    )
                    fig.update_yaxes(
                        row=row_index,
                        col=column_index,
                        visible=False,
                    )
                    fig.add_annotation(
                        text="No data available",
                        row=row_index,
                        col=column_index,
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=14),
                    )

    # Update x and y axes layouts for all subplots
    fig.update_xaxes(
        type="log",
        tickvals=[4096, 2048, 1024, 512, 256, 128, 64, 32, 16],
        ticktext=["4096", "2048", "1024", "512", "256", "128", "64", "32", "16"],
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
        template="plotly_dark",
    )

    # Reduce subplot title font size
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=14)

    # Add a hover template, already shows framework_backend by default
    fig.update_traces(
        hovertemplate=(
            "<b>Batch size:</b>              %{customdata[0]:>9.0f}<br>"
            "<b>Num prompt tokens:</b>       %{x:>9.0f}<br>"
            "<b>Prompt speed (tokens/s):</b> %{customdata[1]:>9.4f}<br>"
            "<b>Gen. speed (tokens/s):</b>   %{customdata[2]:>9.4f}<br>"
            "<b>Prompt time (s):</b>         %{customdata[3]:>9.4f}<br>"
            "<b>Gen. time (s):</b>           %{customdata[4]:>9.4f}<br>"
            "<b>Total time (s):</b>          %{customdata[5]:>9.4f}<br>"
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
