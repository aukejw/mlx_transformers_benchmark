from pathlib import Path
from typing import Dict, Tuple, Union

import fire

import mtb as mtb
from mtb.file_io import aggregate_measurements
from mtb.visualization.create_index import create_index
from mtb.visualization.plot_layer_benchmark_result import show_layer_benchmark_data

DEFAULT_MEASUREMENTS_FOLDER = mtb.REPO_ROOT / "measurements" / "layer_benchmarks"
VISUALIZATIONS_FOLDER = mtb.REPO_ROOT / "visualizations"
OUTPUT_FOLDER = VISUALIZATIONS_FOLDER / "layer_benchmarks"


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

        output_folder_chip = Path(output_folder) / chip_name
        output_folder_chip.mkdir(parents=True, exist_ok=True)

        visualize_chip_measurements(
            measurements_folder=chip_folder,
            output_folder=output_folder_chip,
            show_all_measurements=show_all_measurements,
        )
    return


def visualize_chip_measurements(
    measurements_folder: Path,
    output_folder: Path,
    show_all_measurements: bool,
):
    """Visualize measurements for a specific chip."""
    chip_name = measurements_folder.stem

    relevant_measurements = aggregate_measurements(measurements_folder)
    relevant_measurements = relevant_measurements.sort_values(
        by=["framework_backend", "name", "batch_size", "sequence_length"],
        ignore_index=True,
    )

    benchmark_tasks = sorted(relevant_measurements["name"].unique())

    # Create a mapping from (chip, benchmark) -> results html file
    benchmark_to_figurefile: Dict[Tuple[str, str], Path] = dict()

    print("Visualizing data per benchmark.")
    for benchmark_task in benchmark_tasks:
        relevant_measurements_benchmark = relevant_measurements[
            relevant_measurements.name == benchmark_task
        ]
        print(
            f"  Found {len(relevant_measurements_benchmark):>4} "
            f" datapoints for {benchmark_task}"
        )

        fig = show_layer_benchmark_data(
            title=benchmark_task,
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

        relative_fig_path = fig_path.relative_to(VISUALIZATIONS_FOLDER)
        benchmark_to_figurefile[(chip_name, benchmark_task)] = relative_fig_path

    index_path = create_index(
        visualizations_folder=VISUALIZATIONS_FOLDER,
    )
    print(f"See '{index_path}'")
    return


if __name__ == "__main__":
    fire.Fire(main)
