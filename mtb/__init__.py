import platform
from pathlib import Path

HOME_DIR = Path.home()
REPO_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_HF_HOME = f"{HOME_DIR}/.cache/huggingface/hub"

FLAG_ON_MAC = platform.system() == "Darwin"
FLAG_ON_LINUX = platform.system() == "Linux"
