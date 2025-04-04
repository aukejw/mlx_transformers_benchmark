import platform
import subprocess
import warnings
from typing import Dict

__all__ = [
    "get_mac_hardware_info",
    "get_linux_hardware_info",
]


def get_mac_hardware_info() -> Dict:
    """Get info for this machine, assuming it is a Mac."""
    info = dict(
        processor=platform.processor(),
    )
    try:
        sp_output = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"]
        ).decode("utf-8")

        values_of_interest = dict(
            model_name="Model Name:",
            chip="Chip:",
            total_cores="Total Number of Cores:",
            memory="Memory:",
        )

        for line in sp_output.splitlines():
            for key, lookup in values_of_interest.items():
                if lookup in line:
                    info[key] = line.split(lookup)[1].strip()

    except:
        warnings.warn("Could not obtain hardware information")

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
                mem_gb = round(mem_kb / 1024 / 1024, 2)
                info["memory"] = f"{mem_gb} GB"
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
