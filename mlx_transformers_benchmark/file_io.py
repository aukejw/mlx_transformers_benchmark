import datetime
import warnings
import json
import platform
import subprocess

from typing import Dict, Union
from pathlib import Path


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


def get_mac_hardware_info() -> Dict:
    """Get info for this machine, assuming it is a Mac."""
    info = dict(
        model_name=None,
        chip=None,
        total_cores=None,
        memory=None,
    )
    try:
        sp_output = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"]
        ).decode("utf-8")

        for line in sp_output.splitlines():
            if "Model Name:" in line:
                info["model_name"] = line.split("Model Name:")[1].strip()
            elif "Chip:" in line:
                info["chip"] = line.split("Chip:")[1].strip()
            elif "Total Number of Cores:" in line:
                info["total_cores"] = line.split("Total Number of Cores:")[1].strip()
            elif "Memory:" in line:
                info["memory"] = line.split("Memory:")[1].strip()

    except:
        warnings.warn("Could not obtain hardware information")

    return info
