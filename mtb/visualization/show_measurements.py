from pathlib import Path
from typing import Callable, Union

from natsort import natsort_keygen, natsorted

from mtb.file_io import aggregate_measurements


def show_measurements(
    measurements_folder: Union[Path, str],
    output_folder: Union[Path, str],
    show_all_measurements: bool,
    plot_function: Callable,
    is_llm_benchmark: bool = False,
):
    """Plot measurements in the given folder."""
    relevant_measurements = aggregate_measurements(
        measurements_folder,
        is_llm_benchmark=is_llm_benchmark,
    )

    relevant_measurements = relevant_measurements.sort_values(
        by=["framework_backend", "name", "batch_size", "dtype"],
        key=natsort_keygen(),
        ignore_index=True,
    )

    # Determine how many plots to create
    benchmark_tasks = natsorted(relevant_measurements["name"].unique())
    # force a specific order (high to low precision)
    dtypes = [
        dtype
        for dtype in ("float32", "float16", "bfloat16", "int8", "int4")
        if dtype in set(relevant_measurements["dtype"].unique())
    ]

    print("Visualizing data per benchmark.")
    for benchmark_task in benchmark_tasks:
        relevant_measurements_benchmark = relevant_measurements[
            relevant_measurements.name == benchmark_task
        ]
        print(
            f"  Found {len(relevant_measurements_benchmark):>4} "
            f" datapoints for {benchmark_task}"
        )

        fig = plot_function(
            title=benchmark_task,
            dtypes=dtypes,
            measurements=relevant_measurements_benchmark,
            do_average_measurements=(not show_all_measurements),
        )

        benchmark_shortname = (
            benchmark_task.lower()
            .replace("(", "__")
            .replace(")", "")
            .replace(", ", "_")
        )
        fig_path = output_folder / f"{benchmark_shortname}.html"
        fig.write_html(fig_path)

    return
