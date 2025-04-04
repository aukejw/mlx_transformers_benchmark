import platform
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()

FLAG_ON_MAC = platform.system() == "Darwin"
FLAG_ON_LINUX = platform.system() == "Linux"
