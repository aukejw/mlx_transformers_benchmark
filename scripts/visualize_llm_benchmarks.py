from pathlib import Path
from typing import Union

import fire

import mtb as mtb
from mtb.visualization.create_index import create_index
from mtb.visualization.plot_llm_benchmark_result import show_llm_benchmark_data
from mtb.visualization.show_measurements import show_measurements

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

        show_measurements(
            measurements_folder=chip_folder,
            output_folder=output_folder,
            show_all_measurements=show_all_measurements,
            plot_function=show_llm_benchmark_data,
            is_llm_benchmark=True,
        )

    index_path = create_index(
        visualizations_folder=VISUALIZATIONS_FOLDER,
    )
    print(f"See '{index_path}'")
    return


if __name__ == "__main__":
    fire.Fire(main)
