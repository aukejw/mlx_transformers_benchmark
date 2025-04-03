import datetime
import json
import platform
import subprocess
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from tqdm import tqdm

from mtb.hardware_info import get_mac_hardware_info
from mtb.software_info import get_mlx_version, get_torch_version

__all__ = [
    "create_benchmark_output_dir",
    "create_benchmark_config",
    "aggregate_measurements",
]


def create_benchmark_output_dir(
    output_root: Union[str, Path],
    benchmark_settings: Dict,
) -> Path:
    """Create an output directory for a benchmark run.

    Args:
        output_root: Root directory for benchmark outputs.
        benchmark_settings: Settings to save in the config.

    Returns:
        Output directory.

    """
    configuration: Dict = create_benchmark_config(
        benchmark_settings=benchmark_settings,
    )
    datetime_string = configuration["datetime"]
    hardware_string = configuration["hardware_info"]["chip"].replace(" ", "_")

    output_dir = Path(output_root) / hardware_string / datetime_string
    output_dir.mkdir(parents=True, exist_ok=False)

    with (output_dir / "settings.json").open("w") as f:
        json.dump(configuration, f, indent=2)

    return output_dir


def create_benchmark_config(
    benchmark_settings: Dict,
) -> Dict:
    """Create a configuration that describes the benchmark settings."""

    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d__%H:%M:%S")

    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except:
        git_commit = None

    hardware_info = get_mac_hardware_info()

    software_info = dict(
        platform=platform.platform(),
        python_version=platform.python_version(),
        **get_torch_version(),
        **get_mlx_version(),
    )

    configuration = dict(
        datetime=datetime_string,
        git_commit=git_commit,
        benchmark_settings=benchmark_settings,
        hardware_info=hardware_info,
        software_info=software_info,
    )

    return configuration


def aggregate_measurements(
    measurements_folder: Union[str, Path],
):
    """Collect measurements for the given folder."""
    measurements_folder = Path(measurements_folder)
    measurements_files = measurements_folder.glob("./*/benchmark_results.csv")
    iterator = tqdm(measurements_files, desc="Aggregating measurements..")

    relevant_measurements = []
    for measurements_file in iterator:
        measurements = pd.read_csv(measurements_file)

        settings_file = measurements_file.parent / "settings.json"
        with settings_file.open("r") as f:
            settings = json.load(f)

        # Copy some global settings to the dataframe
        measurements["dtype"] = settings["benchmark_settings"]["dtype"]
        measurements["torch_version"] = settings["software_info"]["torch_version"]
        measurements["mlx_version"] = settings["software_info"]["mlx_version"]

        relevant_measurements.append(measurements)

    relevant_measurements: pd.DataFrame = pd.concat(
        relevant_measurements, ignore_index=True
    )

    relevant_measurements["framework_backend"] = relevant_measurements.apply(
        _convert_row_to_framework_backend,
        axis=1,
    ).astype("category")

    return relevant_measurements


def _convert_row_to_framework_backend(row: pd.Series) -> str:
    """Combine framework and backend into a single string.

    For example:
        framework = 'torch'
        torch_version = '2.0.0'
        backend = 'cpu'

    would result in:
        'torch_2.0.0_cpu'

    """
    name = row["framework"]

    if row["framework"] == "torch":
        name += "_" + row["torch_version"]
    elif row["framework"] == "mlx":
        name += "_" + row["mlx_version"]

    name += "_" + row["backend"]

    if row["compile"]:
        name += "_compiled"

    return name
