import os
from pathlib import Path
from typing import Union

from transformers.utils.logging import disable_progress_bar

hf_home = os.environ.get("HF_HOME", "./hf_cache")


def set_hf_home(path: Union[str, Path] = "./hf_cache"):
    """Set the HF_HOME environment variable to a specific path."""
    global hf_home
    if hf_home != str(path):
        hf_home = str(path)
        os.environ["HF_HOME"] = hf_home
        print(f"HF_HOME set to {hf_home}")

        disable_progress_bar()


def get_hf_home() -> str:
    """Get the HF_HOME environment variable."""
    global hf_home
    return hf_home
