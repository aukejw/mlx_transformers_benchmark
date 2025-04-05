from pathlib import Path
from typing import Dict, Tuple, Union

import fire
from jinja2 import Template

import mtb as mtb
from mtb.file_io import aggregate_measurements
from mtb.visualization.plot_benchmark_result import show_benchmark_data

DEFAULT_MEASUREMENTS_FOLDER = mtb.REPO_ROOT / "measurements" / "Apple_M4_Pro__arm"
VISUALIZATIONS_FOLDER = mtb.REPO_ROOT / "benchmark_visualizations"


def main(
    measurements_folder: Union[str, Path] = DEFAULT_MEASUREMENTS_FOLDER,
):
    """Visualize measurements. We create one page per benchmark task."""

    measurements_folder = Path(measurements_folder)
    chip_name = measurements_folder.stem

    visualizations_folder = VISUALIZATIONS_FOLDER / chip_name
    visualizations_folder.mkdir(parents=True, exist_ok=True)

    relevant_measurements = aggregate_measurements(measurements_folder)
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

        fig = show_benchmark_data(
            title=benchmark_task,
            measurements=relevant_measurements_benchmark,
        )

        benchmark_shortname = (
            benchmark_task.lower()
            .replace("(", "__")
            .replace(")", "")
            .replace(", ", "_")
        )
        fig_path = visualizations_folder / f"{benchmark_shortname}.html"
        fig.write_html(fig_path)

        relative_fig_path = fig_path.relative_to(VISUALIZATIONS_FOLDER)
        benchmark_to_figurefile[(chip_name, benchmark_task)] = relative_fig_path

    create_index(
        visualizations_folder=visualizations_folder,
        benchmark_to_figurefile=benchmark_to_figurefile,
    )
    return


def create_index(
    visualizations_folder: Path,
    benchmark_to_figurefile: Dict[str, Path],
):
    """Create an index file."""

    benchmark_folder = mtb.REPO_ROOT / "benchmark_visualizations"
    template_path = benchmark_folder / "index_template.html"
    with template_path.open() as file:
        template = Template(file.read())

    index_content = template.render(
        visualizations=benchmark_to_figurefile,
    )

    index_path = benchmark_folder / "index.html"
    with index_path.open("w") as f:
        f.write(index_content)

    print(f"See '{index_path}'")
    return


if __name__ == "__main__":
    fire.Fire(main)
