import platform
import re
import subprocess
import warnings
from typing import Dict

__all__ = [
    "get_mac_hardware_info",
    "get_linux_hardware_info",
]


def _find_values_in_string(
    pattern: str,
    string: str,
    default_value: str,
) -> Dict[str, str]:
    """Find the first hit of interest in a string. Return the default value if not found."""

    match = re.search(pattern, string)
    if match:
        return match.group(1).strip()
    else:
        return default_value


def get_mac_hardware_info() -> Dict:
    """Get info for this machine, assuming it is a Mac."""
    info = dict(
        processor=platform.processor(),
    )

    sp_output = subprocess.check_output(
        ["system_profiler", "SPHardwareDataType"]
    ).decode("utf-8")

    display_output = subprocess.check_output(
        ["system_profiler", "SPDisplaysDataType"]
    ).decode("utf-8")

    # Get chip, CPU info
    info["model_name"] = _find_values_in_string(
        pattern=r"Model Name: (.+)",
        string=sp_output,
        default_value="Unknown",
    )
    info["chip"] = _find_values_in_string(
        pattern=r"Chip: (.+)",
        string=sp_output,
        default_value="Unknown",
    )
    info["memory"] = _find_values_in_string(
        pattern=r"Memory: (\d+) GB",
        string=sp_output,
        default_value="X",
    )
    info["total_cores"] = _find_values_in_string(
        pattern=r"Total Number of Cores: (\d+)",
        string=sp_output,
        default_value="X",
    )
    info["performance_cores"] = _find_values_in_string(
        pattern=r"Total Number of Cores: \d+ \((\d+) performance",
        string=sp_output,
        default_value="X",
    )
    info["efficiency_cores"] = _find_values_in_string(
        pattern=r"Total Number of Cores: \d+ \(\d+ performance and (\d+) efficiency",
        string=sp_output,
        default_value="X",
    )

    # Get GPU cores
    info["gpu_cores"] = _find_values_in_string(
        pattern=r"Total Number of Cores: (\d+)",
        string=display_output,
        default_value="X",
    )

    # Summarize machine in one string
    info["hardware_string"] = (
        f"{info['chip']}"
        f"_{info['performance_cores']}P"
        f"+{info['efficiency_cores']}E"
        f"+{info['gpu_cores']}GPU"
        f"_{info['memory']}GB"
    )

    return info


def get_linux_hardware_info() -> Dict:
    """Get info for this machine, assuming it is a Linux system."""
    info = dict(
        processor=None,
        cpu_model=None,
        total_cores=None,
        memory=None,
        chip=None,
    )

    info.update(_get_linux_cpu_info())
    info.update(_get_linux_memory_info())
    info.update(_get_nvidia_info())

    info["hardware_string"] = (
        f"{info['processor']}"
        + (info["chip"] if info["chip"] != "no_gpu" else "")
        + f"_{info['total_cores']}C_{info['memory']}GB"
    )
    return info


def _get_linux_cpu_info() -> Dict:
    """Returns a dict with entries:

    - processor: Processor type (x86_64, arm, etc.)
    - total_cores: Total number of CPU cores

    """
    info = dict()
    lscpu_output = subprocess.check_output(["lscpu"]).decode("utf-8")

    for line in lscpu_output.splitlines():
        if "Architecture:" in line:
            info["processor"] = line.split("Architecture:")[1].strip()

        elif "CPU(s):" in line and "total_cores" not in info:
            info["total_cores"] = line.split("CPU(s):")[1].strip()

    if "processor" not in info:
        raise ValueError(
            "Could not determine processor type! Please check the output "
            "of lscpu on your system."
        )

    return info


def _get_linux_memory_info() -> Dict:
    """Returns a dict with entries:

    - memory: Total RAM in GB

    """
    info = dict()
    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = f.read()

        for line in meminfo.splitlines():
            # Get the total RAM in GB
            if "MemTotal:" in line:
                mem_kb = int(line.split()[1])
                mem_gb = round(mem_kb / 1024**2, 2)
                info["memory"] = f"{mem_gb:.2f}"
                break
    except:
        pass

    if "memory" not in info:
        warnings.warn("Could not obtain memory information")
    return info


def _get_nvidia_info() -> Dict:
    """Returns a dict with entries:

    - chip: GPU name, if available, otherwise "no_gpu"

    """
    info = dict()

    try:
        nvidia_output = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ]
            )
            .decode("utf-8")
            .strip()
        )

        gpu_info = nvidia_output.splitlines()[0].split(", ")
        info["chip"] = gpu_info[0] if len(gpu_info) > 0 else "unknown"
        info["gpu_memory"] = f"{gpu_info[1]} MiB" if len(gpu_info) > 1 else "unknown"
        info["driver_version"] = gpu_info[2] if len(gpu_info) > 2 else "unknown"

    except:
        info["chip"] = "no_gpu"

    return info
