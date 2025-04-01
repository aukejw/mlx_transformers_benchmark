import datetime
import json
import platform
import subprocess
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from tqdm import tqdm

from mlx_transformers_benchmark.platform_info import get_mac_hardware_info


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
    hardware_string = hardware_info["chip"].replace(" ", "_")

    platform_settings = dict(
        platform=platform.platform(),
        processor=platform.processor(),
        python_version=platform.python_version(),
    )

    configuration = dict(
        datetime=datetime_string,
        git_commit=git_commit,
        benchmark_settings=benchmark_settings,
        platform_info=platform_settings,
        hardware_info=hardware_info,
    )

    output_dir = Path(output_root) / hardware_string / datetime_string
    output_dir.mkdir(parents=True, exist_ok=False)

    with (output_dir / "settings.json").open("w") as f:
        json.dump(configuration, f, indent=2)

    return output_dir


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

        measurements["dtype"] = settings["benchmark_settings"]["dtype"]

        relevant_measurements.append(measurements)

    relevant_measurements: pd.DataFrame = pd.concat(
        relevant_measurements, ignore_index=True
    )

    # Combine framework and backend into a single column with unique labels
    relevant_measurements["framework_backend"] = relevant_measurements.apply(
        lambda row: (
            f"{row['framework']}_{row['backend']}"
            + ("_compiled" if row["compile"] is True else "")
        ),
        axis=1,
    ).astype("category")

    return relevant_measurements
