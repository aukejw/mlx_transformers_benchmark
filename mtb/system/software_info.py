import platform
from subprocess import check_output
from typing import Dict

import lmstudio
import mlx.core as mx
import mlx_lm
import torch


def get_software_info() -> Dict:
    """Get an overview of SW info."""
    software_info = dict(
        platform=str(platform.platform()),
        python_version=str(platform.python_version()),
        **get_torch_version(),
        **get_mlx_version(),
        **get_ollama_version(),
    )
    return software_info


def get_torch_version() -> Dict:
    """Get the current torch version and version of important dependencies."""
    return dict(
        torch_version=torch.__version__,
    )


def get_mlx_version() -> Dict:
    """Get the current mlx version and version of important dependencies."""

    return dict(
        mlx_version=mx.__version__,
        mlx_lm_version=mlx_lm.__version__,
    )


def get_lmstudio_version() -> Dict:
    """Get the current lmstudio version and version of important dependencies.

    Note: this is just the version of the python API, and doesn't say anything
    about llama.cpp, LMStudio, or the model files.

    """
    return dict(
        lmstudio_version=lmstudio.__version__,
        # TODO can we determine the runtime version using lms?
        llama_cpp_runtime_version=None,
    )


def get_ollama_version() -> Dict:
    """Get the current ollama version."""
    output = check_output(["ollama", "--version"], text=True)

    # output should be in the form "ollama version is X.Y.Z"
    expected_prefix = "ollama version is "
    if not output.startswith(expected_prefix):
        raise ValueError(f"Unexpected output from 'ollama --version': {output}")

    ollama_version = output.split(expected_prefix)[-1].strip()

    return dict(
        ollama_version=ollama_version,
        # TODO can we determine anything about the llama.cpp engine?
    )
