from unittest.mock import patch

from mtb.hardware_info import get_mac_hardware_info


def test_get_mac_hardware_info_success():
    mock_output = """
    Model Name: MacBook Pro
    Chip: Apple M1
    Total Number of Cores: 8 (4 performance and 4 efficiency)
    Memory: 16 GB
    """
    with patch("subprocess.check_output", return_value=mock_output.encode("utf-8")):
        info = get_mac_hardware_info()
        assert info["model_name"] == "MacBook Pro"
        assert info["chip"] == "Apple M1"
        assert info["total_cores"] == "8"
        assert info["memory"] == "16 GB"
        assert info["processor"] is not None


def test_get_mac_hardware_info_failure():
    with patch("subprocess.check_output", side_effect=Exception("Command failed")):
        info = get_mac_hardware_info()
        assert info["model_name"] is None
        assert info["chip"] is None
        assert info["total_cores"] is None
        assert info["memory"] is None
        assert info["processor"] is not None
