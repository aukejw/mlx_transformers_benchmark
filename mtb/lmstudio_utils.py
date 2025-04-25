import subprocess


def check_lms_server_running() -> bool:
    """Return if the LM Studio server is running."""

    command = ["lms", "server", "status"]
    try:
        response = (
            subprocess.check_output(command, stderr=subprocess.STDOUT)
            .decode("utf-8")
            .strip()
        )
    except FileNotFoundError:
        # Likely, lms is not available
        return False

    success_message = "The server is running on port"
    if response.startswith(success_message):
        return True
    else:
        return False
