from pathlib import Path
from typing import Union

import fire

import mlx_transformers_benchmark as mtb
from mlx_transformers_benchmark.file_io import aggregate_measurements
from mlx_transformers_benchmark.visualization.plot_benchmark_result import (
    show_benchmark_data,
)

DEFAULT_MEASUREMENTS_FOLDER = mtb.REPO_ROOT / "measurements" / "Apple_M4_Pro"
DEFAULT_VISUALIZATIONS_FOLDER = mtb.REPO_ROOT / "benchmark_visualizations"


def main(
    measurements_folder: Union[str, Path] = DEFAULT_MEASUREMENTS_FOLDER,
    visualizations_folder: Union[str, Path] = DEFAULT_VISUALIZATIONS_FOLDER,
):
    """Visualize measurements. We create one page per benchmark task."""

    measurements_folder = Path(measurements_folder)
    visualizations_folder = Path(visualizations_folder)
    visualizations_folder.mkdir(parents=True, exist_ok=True)

    relevant_measurements = aggregate_measurements(measurements_folder)
    benchmark_tasks = sorted(relevant_measurements["name"].unique())

    print("Visualizing data per benchmark.")
    for benchmark_task in benchmark_tasks:
        relevant_measurements_benchmark = relevant_measurements[
            relevant_measurements.name == benchmark_task
        ]
        print(
            f"  Found {len(relevant_measurements_benchmark):>4} "
            f" datapoints for {benchmark_task}"
        )

        fig = show_benchmark_data(
            title=benchmark_task,
            measurements=relevant_measurements_benchmark,
        )

        benchmark_shortname = (
            benchmark_shortname.lower()
            .replace("(", "__")
            .replace(")", "")
            .replace(", ", "_")
        )
        fig.write_html(visualizations_folder / f"{benchmark_shortname}.html")
        fig.show()

    return


if __name__ == "__main__":
    fire.Fire(main)
