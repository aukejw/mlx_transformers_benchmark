from pathlib import Path
from typing import Dict, Union

from jinja2 import Template

import mtb

VISUALIZATIONS_FOLDER = mtb.REPO_ROOT / "visualizations"


def create_index(
    output_folder: Union[str, Path],
    benchmark_to_figurefile: Dict[str, Path],
):
    """Create an index file.

    Args:
        benmark_to_figurefile: Mapping from benchmark name to the path
            of a benchmark visualization html file.

    Returns:
        Path to the index.html file.

    """
    template_path = VISUALIZATIONS_FOLDER / "index_template.html"
    with template_path.open() as file:
        template = Template(file.read())

    index_content = template.render(
        visualizations=benchmark_to_figurefile,
    )

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    index_path = output_folder / "index.html"
    with index_path.open("w") as f:
        f.write(index_content)

    return index_path
