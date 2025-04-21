import json

import pandas as pd
import pytest

from mtb.visualization.plot_layer_benchmark_result import show_layer_benchmark_data
from mtb.visualization.plot_llm_benchmark_result import show_llm_benchmark_data
from mtb.visualization.show_measurements import show_measurements


@pytest.fixture
def settings():
    return dict(
        datetime="2023-10-01__12:00:00",
        git_commit="dummy_commit",
        benchmark_settings=dict(
            num_iterations=10,
            num_warmup_iterations=5,
        ),
        hardware_info=dict(
            hardware_string="Apple_M4",
            hardware_type="CPU",
        ),
        software_info=dict(
            platform="macOS",
            python_version="3.11.11",
            torch_version="2.0.0",
            mlx_version="1.0.0",
            mlx_lm_version="1.0.0",
        ),
    )


@pytest.fixture
def layer_measurements_folder(tmp_path, settings):
    folder = tmp_path / "measurements" / "layer_benchmarks" / "Apple_M4"
    folder.mkdir(parents=True, exist_ok=True)

    # Create dummy measurements files
    dummy_file = folder / "benchmark_results.csv"
    dataframe = pd.DataFrame(
        dict(
            framework=["torch", "mlx"],
            backend=["mps", "metal"],
            name=["MhsaBenchmark(dim=10)", "MhsaBenchmark(dim=10)"],
            batch_size=[32, 64],
            dtype=["float32", "float16"],
            sequence_length=[128, 256],
            duration_ms=[1.0, 2.0],
        )
    )
    dataframe.to_csv(dummy_file, index=False)

    settings_file = folder / "settings.json"
    with settings_file.open("w") as f:
        json.dump(settings, f, indent=2)

    return folder.parent


@pytest.fixture
def llm_measurements_folder(tmp_path, settings):
    folder = tmp_path / "measurements" / "llm_benchmarks" / "Apple_M4"
    folder.mkdir(parents=True, exist_ok=True)

    # Create dummy measurements files
    dummy_file = folder / "benchmark_results.csv"
    dataframe = pd.DataFrame(
        dict(
            framework=["torch", "mlx"],
            backend=["mps", "metal"],
            name=["gemma-3-1b-it", "gemma-3-1b-it"],
            batch_size=[32, 64],
            dtype=["float32", "float16"],
            num_prompt_tokens=[128, 256],
            num_generated_tokens=[100, 100],
            prompt_time_sec=[1.0, 2.0],
            prompt_tps=[32.0, 64.0],
            generation_time_sec=[1.0, 2.0],
            generation_tps=[32.0, 64.0],
        )
    )
    dataframe.to_csv(dummy_file, index=False)

    settings_file = folder / "settings.json"
    with settings_file.open("w") as f:
        json.dump(settings, f, indent=2)

    return folder.parent


def test_show_layer_measurements(layer_measurements_folder, tmp_path):
    output_folder = tmp_path / "output"
    plot_function = show_layer_benchmark_data
    show_measurements(
        measurements_folder=layer_measurements_folder,
        output_folder=output_folder,
        show_all_measurements=False,
        plot_function=plot_function,
        is_llm_benchmark=False,
    )
    assert (output_folder / "mhsabenchmark__dim=10.html").exists()


def test_show_llm_measurements(llm_measurements_folder, tmp_path):
    output_folder = tmp_path / "output"
    plot_function = show_llm_benchmark_data
    show_measurements(
        measurements_folder=llm_measurements_folder,
        output_folder=output_folder,
        show_all_measurements=True,
        plot_function=plot_function,
        is_llm_benchmark=True,
    )
    assert (output_folder / "gemma-3-1b-it.html").exists()
