import json

import pandas as pd
import pytest

from mtb.file_io import (
    _convert_row_to_framework_backend,
    _get_commit,
    aggregate_measurements,
    create_benchmark_config,
    create_benchmark_output_dir,
)

mlx_version_return_value = {
    "mlx_version": "1.0.0",
    "mlx_lm_version": "1.0.0",
}

torch_version_return_value = {
    "torch_version": "2.0.0",
}


@pytest.fixture
def benchmark_settings():
    return {"dtype": "float32", "num_iterations": 10, "num_repeats": 3}


@pytest.fixture
def output_root(tmp_path):
    return tmp_path


def test_create_benchmark_output_dir(output_root, benchmark_settings):
    output_dir = create_benchmark_output_dir(output_root, benchmark_settings)
    assert output_dir.exists()
    assert (output_dir / "settings.json").exists()

    with (output_dir / "settings.json").open("r") as f:
        settings = json.load(f)
        assert settings["benchmark_settings"] == benchmark_settings


def test_create_benchmark_config_mac(mocker, benchmark_settings):
    mocker.patch("platform.system", return_value="Darwin")
    mocker.patch(
        "mtb.file_io.get_hardware_info",
        return_value={"chip": "M1"},
    )
    mocker.patch(
        "mtb.system.software_info.get_mlx_version",
        return_value=mlx_version_return_value,
    )
    mocker.patch(
        "mtb.system.software_info.get_torch_version",
        return_value=torch_version_return_value,
    )

    config = create_benchmark_config(benchmark_settings)
    assert config["benchmark_settings"] == benchmark_settings
    assert config["software_info"]["mlx_version"] == "1.0.0"
    assert config["software_info"]["mlx_lm_version"] == "1.0.0"
    assert config["software_info"]["torch_version"] == "2.0.0"
    assert config["hardware_info"]["chip"] == "M1"


def test_create_benchmark_config_linux(mocker, benchmark_settings):
    mocker.patch("platform.system", return_value="Linux")
    mocker.patch(
        "mtb.system.hardware_info.get_linux_hardware_info",
        return_value={"chip": "aarch64"},
    )
    mocker.patch(
        "mtb.system.software_info.get_mlx_version",
        return_value=mlx_version_return_value,
    )
    mocker.patch(
        "mtb.system.software_info.get_torch_version",
        return_value=torch_version_return_value,
    )

    config = create_benchmark_config(benchmark_settings)
    assert config["benchmark_settings"] == benchmark_settings
    assert config["hardware_info"]["chip"] == "aarch64"
    assert config["software_info"]["mlx_version"] == "1.0.0"
    assert config["software_info"]["mlx_lm_version"] == "1.0.0"
    assert config["software_info"]["torch_version"] == "2.0.0"


def test_get_commit(mocker):
    mocker.patch("mtb.file_io.subprocess.check_output", return_value=b"mock_git_commit")
    commit = _get_commit()
    assert commit == "mock_git_commit"


def test_get_illegal_commit(mocker):
    mocker.patch(
        "mtb.file_io.subprocess.check_output", side_effect=Exception("Command failed")
    )
    commit = _get_commit()
    assert commit == "commit_unknown"


def test_aggregate_measurements(tmp_path):
    # Create a mock measurements directory
    measurements_dir = tmp_path / "measurements"
    run_dir = measurements_dir / "measurement1"
    run_dir.mkdir(parents=True)

    # Create a mock CSV file
    csv_file = run_dir / "benchmark_results.csv"
    mock_data = {
        "name": ["SomeBenchmark"],
        "framework": ["torch"],
        "backend": ["cpu"],
        "batch_size": [1],
        "dtype": "bfloat16",
        "sequence_length": [16],
        "compile": [False],
        "mean_ms": [1.5],
    }
    pd.DataFrame(mock_data).to_csv(csv_file, index=False)

    # Create a mock settings.json file
    settings_file = run_dir / "settings.json"
    mock_settings = {
        "benchmark_settings": {
            "num_warmup_iterations": 5,
            "num_iterations": 10,
            "num_repeats": 3,
        },
        "software_info": {
            "torch_version": "2.0.0",
            "mlx_version": "1.0.0",
        },
    }
    with settings_file.open("w") as f:
        json.dump(mock_settings, f)

    result = aggregate_measurements(measurements_dir)

    assert "dtype" in result.columns
    assert "torch_version" in result.columns
    assert "mlx_version" in result.columns
    assert result.iloc[0]["framework"] == "torch"
    assert result.iloc[0]["backend"] == "cpu"
    assert result.iloc[0]["compile"] == False
    assert result.iloc[0]["mean_ms"] == 1.5


def test_convert_row_to_framework_backend():
    series = pd.Series(
        data=dict(
            framework="torch",
            torch_version="2.0.0",
            backend="cpu",
            compile=False,
        )
    )
    name = _convert_row_to_framework_backend(series)
    assert name == "torch_2.0.0_cpu"

    series = pd.Series(
        data=dict(
            framework="mlx",
            mlx_version="0.24.1",
            backend="metal",
            compile=True,
        )
    )
    name = _convert_row_to_framework_backend(series)
    assert name == "mlx_0.24.1_metal_compiled"

    series = pd.Series(
        data=dict(
            framework="illegal",
            mlx_version="0.24.1",
            backend="metal",
            compile=True,
        )
    )
    with pytest.raises(NotImplementedError):
        _convert_row_to_framework_backend(series)
