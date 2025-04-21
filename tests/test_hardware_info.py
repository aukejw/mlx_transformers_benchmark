import platform
from unittest.mock import patch

import pytest

from mtb.hardware_info import (
    _get_linux_cpu_info,
    _get_linux_memory_info,
    _get_nvidia_info,
    get_linux_hardware_info,
    get_mac_hardware_info,
)


def test_get_mac_hardware_info_success():
    mock_output = """
    Model Name: MacBook Pro
    Chip: Apple M1
    Total Number of Cores: 8 (4 performance and 4 efficiency)
    Memory: 16 GB
    """
    with patch("subprocess.check_output", return_value=mock_output.encode("utf-8")):
        info = get_mac_hardware_info()
        print(info)
        assert info["model_name"] == "MacBook Pro"
        assert info["chip"] == "Apple M1"
        assert info["total_cores"] == "8"
        assert info["performance_cores"] == "4"
        assert info["efficiency_cores"] == "4"
        assert info["gpu_cores"] == "8"
        assert info["memory"] == "16"
        assert info["hardware_string"] == "Apple M1_4P+4E+8GPU_16GB"
        assert info["processor"] is not None


@pytest.mark.skipif(platform.system() != "Linux", reason="platform != Linux")
def test_get_linux_hardware_info():
    info = get_linux_hardware_info()
    assert "processor" in info
    assert "cpu_model" in info
    assert "total_cores" in info
    assert "memory" in info

    assert info["hardware_string"] == (
        f"{info['processor']}_{info['cpu_model']}_{info['total_cores']}C_{info['memory']}GB"
    )


@patch("mtb.hardware_info._get_linux_cpu_info", return_value=dict(processor="aarch64"))
@patch("mtb.hardware_info._get_linux_memory_info", return_value=dict(memory="1.00 GB"))
@patch(
    "mtb.hardware_info._get_nvidia_info",
    return_value=dict(chip="Nvidia GeForce RTX 3080"),
)
def test_get_linux_hardware_info_mocked(
    mock_cpu_info, mock_memory_info, mock_nvidia_info
):
    # bit of a useless test, but cannot run on mac otherwise
    info = get_linux_hardware_info()
    assert info["processor"] == "aarch64"
    assert info["memory"] == "1.00"
    assert info["chip"] == "Nvidia GeForce RTX 3080"


@patch("platform.processor", return_value="arm")
@patch("subprocess.check_output", side_effect=Exception("Command failed"))
def test_get_mac_hardware_info_failure(mock_processor, mock_check_output):
    info = get_mac_hardware_info()
    assert info == dict(processor="arm")


@patch("platform.processor", return_value="x86_64")
@pytest.mark.parametrize("processor", ["x86_64", "aarch64"])
def test_get_linux_cpu_info(mock_platform, processor):
    mock_output = f"""
    Architecture: {processor}
    CPU(s): 8
    """
    with patch("subprocess.check_output", return_value=mock_output.encode("utf-8")):
        info = _get_linux_cpu_info()
        assert info["processor"] == processor
        assert info["total_cores"] == "8"


@patch("platform.processor", return_value="x86_64")
def test_get_linux_cpu_info_noprocessor(mock_platform):
    mock_output = f"""
    CPU(s): 8
    """
    with patch("subprocess.check_output", return_value=mock_output.encode("utf-8")):
        with pytest.raises(ValueError):
            _get_linux_cpu_info()


@patch("builtins.open")
def test_get_linux_memory_info_success(mock_open, tmp_path):
    mock_file_content = """
    MemTotal:       32723584 kB
    MemFree:        22498452 kB
    """
    fake_proc_info = tmp_path / "meminfo"
    fake_proc_info.write_text(mock_file_content)
    mock_open.return_value.__enter__.return_value = fake_proc_info.open("r")

    info = _get_linux_memory_info()
    assert info["memory"] == "31.21 GB"


@patch("builtins.open")
def test_get_linux_memory_info_missing(mock_open, tmp_path):
    mock_file_content = """
    MemFree:        22498452 kB
    """
    fake_proc_info = tmp_path / "meminfo"
    fake_proc_info.write_text(mock_file_content)
    mock_open.return_value.__enter__.return_value = fake_proc_info.open("r")

    with pytest.warns():
        info = _get_linux_memory_info()
    assert "memory" not in info


@patch("builtins.open", side_effect=ValueError)
def test_get_linux_memory_info_missing_data(mock_open, tmp_path):
    with pytest.warns():
        info = _get_linux_memory_info()
    assert "memory" not in info


@patch("subprocess.check_output")
def test_get_nvidia_info_success(mock_check_output):
    mock_output = "Nvidia GeForce RTX 3080, 10240, 460.32.03"
    mock_check_output.return_value = mock_output.encode("utf-8")

    info = _get_nvidia_info()
    assert info["chip"] == "Nvidia GeForce RTX 3080"
    assert info["gpu_memory"] == "10240 MiB"
    assert info["driver_version"] == "460.32.03"


@patch("subprocess.check_output", side_effect=Exception("Command failed"))
def test_get_nvidia_info_failure(mock_check_output):
    info = _get_nvidia_info()
    assert info == dict(chip="no_gpu")
