from mtb.software_info import get_mlx_version, get_torch_version


def test_get_torch_version():
    result = get_torch_version()
    assert "torch_version" in result
    assert isinstance(result["torch_version"], str)
    assert result["torch_version"] != ""


def test_get_mlx_version():
    result = get_mlx_version()
    assert "mlx_version" in result
    assert isinstance(result["mlx_version"], str)
    assert result["mlx_version"] != ""
