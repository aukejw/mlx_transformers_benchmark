from pathlib import Path
from typing import Union

from jinja2 import Template

import mtb

VISUALIZATIONS_FOLDER = mtb.REPO_ROOT / "visualizations"


def create_index(
    visualizations_folder: Union[str, Path],
):
    """Create an index file.

    Args:
        template_path: Path to Jinja2 template file.
        visualizations_folder: Path to folder containing the viz html files. Structure
            is assumed to be `./<benchmark_type>/<chip_name>/<benchmark_name>.html`.
        output_folder: Path where index will be created.

    Returns:
        Path to the index.html file.

    """
    visualizations_folder = Path(visualizations_folder)

    template_path = visualizations_folder / "index_template.html"
    with Path(template_path).open("r") as file:
        template = Template(file.read())

    # Create a mapping from (benchmark_type chip, benchmark) -> html file
    benchmark_to_figurefile = dict()
    for benchmark_file in sorted(visualizations_folder.glob("./*/*/*.html")):
        benchmark_file = benchmark_file.relative_to(visualizations_folder)
        benchmark_type = benchmark_file.parts[0]
        chip_name = benchmark_file.parts[1]
        benchmark_name = benchmark_file.stem

        key = (benchmark_type, chip_name, benchmark_name)
        benchmark_to_figurefile[key] = benchmark_file.as_posix()

    # Create the index content
    index_content = template.render(
        visualizations=benchmark_to_figurefile,
    )

    # Write to file, save next to index_template
    output_folder = Path(template_path).parent
    index_path = output_folder / "index.html"
    with index_path.open("w") as f:
        f.write(index_content)

    return index_path
