import os
from pathlib import Path
from typing import Union

from huggingface_hub import snapshot_download
from transformers.utils.logging import disable_progress_bar, enable_progress_bar

from mtb import DEFAULT_HF_HOME

hf_home = os.environ.get("HF_HOME", DEFAULT_HF_HOME)


def set_hf_home(
    path: Union[str, Path] = DEFAULT_HF_HOME,
    enable_hf_progressbar: bool = False,
):
    """Set the HF_HOME environment variable to a specific path.

    By default, we also disable the progress bar.

    """
    global hf_home

    if hf_home != str(path):
        hf_home = str(path)
        os.environ["HF_HOME"] = hf_home
        print(f"HF_HOME set to '{hf_home}' ")

    if not enable_hf_progressbar:
        disable_progress_bar()


def get_hf_home() -> str:
    """Get the HF_HOME environment variable."""
    global hf_home
    return hf_home


def verbose_download_model(
    model_id: str,
    **kwargs,
):
    """Download a model from Hugging Face with verbose output."""
    # Check if the model is already downloaded
    try:
        snapshot_folder = snapshot_download(
            model_id,
            local_files_only=True,
            cache_dir=get_hf_home(),
            **kwargs,
        )
    except Exception as e:
        print(f"\nDownloading model '{model_id}'...\n")

        enable_progress_bar()
        snapshot_folder = snapshot_download(
            model_id,
            local_files_only=False,
            cache_dir=get_hf_home(),
            **kwargs,
        )
        disable_progress_bar()
        print(f"\n\nDownloaded model '{model_id}'.")

    return snapshot_folder
