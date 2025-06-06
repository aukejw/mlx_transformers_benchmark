import datetime
import json
import subprocess
from functools import partial
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from tqdm import tqdm

from mtb.system.hardware_info import get_hardware_info
from mtb.system.software_info import get_software_info

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
    output_root = Path(output_root)

    configuration: Dict = create_benchmark_config(
        benchmark_settings=benchmark_settings,
    )

    datetime_string = configuration["datetime"]
    hardware_string = configuration["hardware_info"]["hardware_string"]

    output_dir = output_root / hardware_string / datetime_string
    output_dir.mkdir(parents=True, exist_ok=False)

    with (output_dir / "settings.json").open("w") as f:
        json.dump(configuration, f, indent=2)

    return output_dir


def create_benchmark_config(
    benchmark_settings: Dict,
) -> Dict:
    """Create a configuration that describes the benchmark settings."""

    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d__%H:%M:%S")

    configuration = dict(
        datetime=datetime_string,
        contributor="",
        git_commit=_get_commit(),
        benchmark_settings=benchmark_settings,
        hardware_info=get_hardware_info(),
        software_info=get_software_info(),
    )

    return configuration


def _get_commit() -> str:
    """Get the current git commit hash."""
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except:
        git_commit = "commit_unknown"
    return git_commit


def aggregate_measurements(
    measurements_folder: Union[str, Path],
    is_llm_benchmark: bool = False,
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
        for key in [
            "num_warmup_iterations",
            "num_iterations",
        ]:
            measurements[key] = settings["benchmark_settings"][key]

        for key in [
            "torch_version",
            "mlx_version",
            "mlx_lm_version",
        ]:
            measurements[key] = settings["software_info"].get(key, None)

        relevant_measurements.append(measurements)

    relevant_measurements: pd.DataFrame = pd.concat(
        relevant_measurements, ignore_index=True
    )

    relevant_measurements["framework_backend"] = relevant_measurements.apply(
        partial(_convert_row_to_framework_backend, is_llm_benchmark=is_llm_benchmark),
        axis=1,
    ).astype("category")

    # add columns of interest combining existing ones
    if is_llm_benchmark and "total_time_sec" not in relevant_measurements.columns:
        relevant_measurements["total_time_sec"] = (
            relevant_measurements["prompt_time_sec"]
            + relevant_measurements["generation_time_sec"]
        )

    return relevant_measurements


def _convert_row_to_framework_backend(
    row: pd.Series,
    is_llm_benchmark: bool = False,
) -> str:
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
    elif row["framework"] == "lmstudio":
        pass
    elif row["framework"] == "lmstudio_mlx":
        name = "lmstudio"
    elif row["framework"] == "ollama":
        pass
    else:
        raise NotImplementedError(f"Unsupported framework {row['framework']}")

    name += "_" + row["backend"]

    if is_llm_benchmark and row["framework"] == "mlx":
        name += "__mlx_lm_" + row["mlx_lm_version"]

    if row.get("compile", False):
        name += "_compiled"

    return name
