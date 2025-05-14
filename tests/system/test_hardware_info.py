import platform

import pytest

from mtb.system.hardware_info import (
    _get_linux_cpu_info,
    _get_linux_memory_info,
    _get_nvidia_info,
    get_hardware_info,
    get_linux_hardware_info,
    get_mac_hardware_info,
)


def test_get_hardware_info_mac(mocker):
    mocker.patch(
        "mtb.system.hardware_info.platform.system",
        return_value="Darwin",
    )
    mocker.patch(
        "mtb.system.hardware_info.get_mac_hardware_info",
        return_value={"chip": "Apple M1"},
    )
    info = get_hardware_info()
    assert info["chip"] == "Apple M1"


def test_get_hardware_info_linux(mocker):
    mocker.patch(
        "mtb.system.hardware_info.platform.system",
        return_value="Linux",
    )
    mocker.patch(
        "mtb.system.hardware_info.get_linux_hardware_info",
        return_value={"chip": "Nvidia GeForce RTX 3080"},
    )
    info = get_hardware_info()
    assert info["chip"] == "Nvidia GeForce RTX 3080"


def test_get_hardware_info_unsupported(mocker):
    mocker.patch(
        "mtb.system.hardware_info.platform.system", return_value="illegal_value"
    )
    with pytest.raises(NotImplementedError):
        get_hardware_info()


def test_get_mac_hardware_info_success(mocker):
    mock_check_output = mocker.patch("mtb.system.hardware_info.check_output")
    mock_output = """
    Model Name: MacBook Pro
    Chip: Apple M1
    Total Number of Cores: 8 (4 performance and 4 efficiency)
    Memory: 16 GB
    """
    mock_check_output.return_value = mock_output.encode("utf-8")

    info = get_mac_hardware_info()
    assert info["model_name"] == "MacBook Pro"
    assert info["chip"] == "Apple M1"
    assert info["total_cores"] == "8"
    assert info["performance_cores"] == "4"
    assert info["efficiency_cores"] == "4"
    assert info["gpu_cores"] == "8"
    assert info["memory"] == "16"
    assert info["hardware_string"] == "Apple_M1_4P+4E+8GPU_16GB"
    assert info["processor"] is not None


@pytest.mark.skipif(platform.system() != "Linux", reason="platform != Linux")
def test_get_linux_hardware_info():
    info = get_linux_hardware_info()
    assert "processor" in info
    assert "cpu_model" in info
    assert "chip" in info
    assert "total_cores" in info
    assert "memory" in info

    expected_hardware_string = (
        f"{info['processor']}"
        + (f"_{info['chip']}" if info["chip"] != "no_gpu" else "")
        + f"_{info['total_cores']}C_{info['memory']}GB"
    )
    assert info["hardware_string"] == expected_hardware_string


def test_get_linux_hardware_info_mocked(mocker):
    mocker.patch(
        "mtb.system.hardware_info._get_linux_cpu_info",
        return_value=dict(processor="aarch64"),
    )
    mocker.patch(
        "mtb.system.hardware_info._get_linux_memory_info",
        return_value=dict(memory="1.00"),
    )
    mocker.patch(
        "mtb.system.hardware_info._get_nvidia_info",
        return_value=dict(chip="Nvidia GeForce RTX 3080"),
    )
    # bit of a useless test, but cannot run on mac otherwise
    info = get_linux_hardware_info()
    assert info["processor"] == "aarch64"
    assert info["memory"] == "1.00"
    assert info["chip"] == "Nvidia GeForce RTX 3080"


def test_get_mac_hardware_info_failure(mocker):
    mocker.patch(
        "mtb.system.hardware_info.platform.processor",
        return_value="arm",
    )
    mocker.patch(
        "mtb.system.hardware_info.check_output",
        return_value=b"illegal_value",
    )
    info = get_mac_hardware_info()
    assert info["processor"] == "arm"
    assert info["chip"] == "Unknown"
    assert info["total_cores"] == "X"
    assert info["performance_cores"] == "X"
    assert info["efficiency_cores"] == "X"
    assert info["memory"] == "X"


@pytest.mark.parametrize("processor", ["x86_64", "aarch64"])
def test_get_linux_cpu_info(mocker, processor):
    mocker.patch("mtb.system.hardware_info.platform.processor", return_value=processor)
    mock_check_output = mocker.patch("mtb.system.hardware_info.check_output")
    mock_output = f"""
    Architecture: {processor}
    CPU(s): 8
    """
    mock_check_output.return_value = mock_output.encode("utf-8")

    info = _get_linux_cpu_info()
    assert info["processor"] == processor
    assert info["total_cores"] == "8"


def test_get_linux_cpu_info_noprocessor(mocker):
    mocker.patch("mtb.system.hardware_info.platform.processor", return_value="x86_64")
    mock_check_output = mocker.patch("mtb.system.hardware_info.check_output")
    mock_output = f"""
    CPU(s): 8
    """
    mock_check_output.return_value = mock_output.encode("utf-8")
    with pytest.raises(ValueError):
        _get_linux_cpu_info()


def test_get_linux_memory_info_success(mocker, tmp_path):
    mock_open = mocker.patch("builtins.open")
    mock_file_content = """
    MemTotal:       32723584 kB
    MemFree:        22498452 kB
    """
    fake_proc_info = tmp_path / "meminfo"
    fake_proc_info.write_text(mock_file_content)
    mock_open.return_value.__enter__.return_value = fake_proc_info.open("r")

    info = _get_linux_memory_info()
    assert info["memory"] == "32.72"


def test_get_linux_memory_info_missing(mocker, tmp_path):
    mock_open = mocker.patch("builtins.open")
    mock_file_content = """
    MemFree:        22498452 kB
    """
    fake_proc_info = tmp_path / "meminfo"
    fake_proc_info.write_text(mock_file_content)
    mock_open.return_value.__enter__.return_value = fake_proc_info.open("r")

    with pytest.warns():
        info = _get_linux_memory_info()
    assert "memory" not in info


def test_get_linux_memory_info_missing_data(mocker):
    mocker.patch("builtins.open", side_effect=ValueError)
    with pytest.warns():
        info = _get_linux_memory_info()
    assert "memory" not in info


def test_get_nvidia_info_success(mocker):
    mock_check_output = mocker.patch("mtb.system.hardware_info.check_output")
    mock_output = "Nvidia GeForce RTX 3080, 10240, 460.32.03"
    mock_check_output.return_value = mock_output.encode("utf-8")

    info = _get_nvidia_info()
    assert info["chip"] == "Nvidia GeForce RTX 3080"
    assert info["gpu_memory"] == "10240 MiB"
    assert info["driver_version"] == "460.32.03"


def test_get_nvidia_info_failure(mocker):
    mocker.patch(
        "mtb.system.hardware_info.check_output",
        side_effect=Exception("Command failed"),
    )
    info = _get_nvidia_info()
    assert info == dict(chip="no_gpu")
