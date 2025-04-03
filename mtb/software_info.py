from typing import Dict

import mlx.core as mx
import torch


def get_torch_version() -> Dict:
    """Get the current torch version and version of important dependencies."""
    return dict(
        torch_version=torch.__version__,
    )


def get_mlx_version() -> Dict:
    """Get the current mlx version and version of important dependencies."""

    return dict(
        mlx_version=mx.__version__,
    )
