from pathlib import Path
from typing import Union

import fire
from natsort import natsort_keygen, natsorted

import mtb as mtb
import mtb.visualization.symbol_and_color
from mtb.file_io import aggregate_measurements
from mtb.visualization.create_index import create_index
from mtb.visualization.plot_llm_benchmark_result import show_llm_benchmark_data

DEFAULT_MEASUREMENTS_FOLDER = mtb.REPO_ROOT / "measurements" / "llm_benchmarks"
VISUALIZATIONS_FOLDER = mtb.REPO_ROOT / "visualizations"
OUTPUT_FOLDER = VISUALIZATIONS_FOLDER / "llm_benchmarks"


def main(
    measurements_folder: Union[str, Path] = DEFAULT_MEASUREMENTS_FOLDER,
    show_all_measurements: bool = False,
    output_folder: Union[str, Path] = OUTPUT_FOLDER,
):
    """Visualize measurements. We create one page per benchmark task.

    Args:
        measurements_folder: Folder containing the measurements.
        show_all_measurements: If True, show all individual measurements.
        output_folder: Folder to save the visualizations in.

    """
    measurements_folder = Path(measurements_folder)
    for chip_folder in sorted(measurements_folder.glob("*")):
        chip_name = chip_folder.stem
        output_folder = Path(output_folder) / chip_name
        output_folder.mkdir(parents=True, exist_ok=True)

        visualize_chip_measurements(
            measurements_folder=chip_folder,
            output_folder=output_folder,
            show_all_measurements=show_all_measurements,
        )

    index_path = create_index(
        visualizations_folder=VISUALIZATIONS_FOLDER,
    )
    print(f"See '{index_path}'")
    return


def visualize_chip_measurements(
    measurements_folder: Path,
    output_folder: Path,
    show_all_measurements: bool,
):
    """Visualize measurements for a specific chip."""

    relevant_measurements = aggregate_measurements(
        measurements_folder,
        is_llm_benchmark=True,
    )

    # add columns of interest combining existing ones
    relevant_measurements["total_time_sec"] = (
        relevant_measurements["prompt_time_sec"]
        + relevant_measurements["generation_time_sec"]
    )
    relevant_measurements = relevant_measurements.sort_values(
        by=["framework_backend", "name", "batch_size", "dtype", "num_prompt_tokens"],
        key=natsort_keygen(),
        ignore_index=True,
    )

    # Filter out measurements with no prompt time
    benchmark_tasks = natsorted(relevant_measurements["name"].unique())
    dtypes = [
        dtype
        for dtype in ("float32", "float16", "bfloat16", "int8", "int4")
        if dtype in set(relevant_measurements["dtype"].unique())
    ]

    # Define colors, symbols
    frameworks = natsorted(relevant_measurements["framework_backend"].unique())
    for framework in frameworks:
        mtb.visualization.symbol_and_color.add_category(framework)

    print("Visualizing data per benchmark.")
    for benchmark_task in benchmark_tasks:
        relevant_measurements_benchmark = relevant_measurements[
            relevant_measurements.name == benchmark_task
        ]
        print(
            f"  Found {len(relevant_measurements_benchmark):>4} "
            f" datapoints for {benchmark_task}"
        )

        fig = show_llm_benchmark_data(
            title=benchmark_task,
            dtypes=dtypes,
            frameworks=frameworks,
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


if __name__ == "__main__":
    fire.Fire(main)
