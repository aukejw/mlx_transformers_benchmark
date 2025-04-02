import os
import platform
import subprocess
import warnings
from pathlib import Path
from typing import Dict, Union


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
