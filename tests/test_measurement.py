import pytest

from mtb.measurement import LlmBenchmarkMeasurement, Measurements


def test_llm_benchmark_measurement():
    meausrement = LlmBenchmarkMeasurement(
        response="Hello, world!",
        prompt_time_sec=0.1,
        prompt_tps=10.0,
        generation_time_sec=0.2,
        generation_tps=20.0,
        num_prompt_tokens=5,
        num_generated_tokens=10,
    )
    assert meausrement.response == "Hello, world!"
    assert meausrement.to_dict(include_reponse=True) == {
        "response": "Hello, world!",
        "prompt_time_sec": 0.1,
        "prompt_tps": 10.0,
        "generation_time_sec": 0.2,
        "generation_tps": 20.0,
        "num_prompt_tokens": 5,
        "num_generated_tokens": 10,
    }


def test_measurements():
    measurements = Measurements()
    assert measurements.keys == []

    measurements.add(
        {
            "runtime": 0.1,
            "memory": 10.0,
        }
    )
    measurements.add(
        {
            "runtime": 0.3,
            "memory": 20.0,
        }
    )

    assert measurements._num_measurements == 2
    assert measurements.get_mean("runtime") == 0.2
    assert measurements.get_mean("memory") == 15.0

    expected_means = {
        "runtime": 0.2,
        "memory": 15.0,
    }
    assert measurements.get_means() == expected_means

    assert str(measurements) == (
        "Measurements("
        "\n  num_measurements=2,"
        "\n  runtime=0.2000,"
        "\n  memory=15.0000,"
        "\n)"
    )

    measurements.reset()
    assert measurements._num_measurements == 0


def test_measurements_errors():
    measurements = Measurements()
    measurements.add(
        {
            "runtime": 0.1,
            "memory": 10.0,
        }
    )

    # incomplete entry
    with pytest.raises(KeyError):
        measurements.add({"runtime": 0.2})

    # key not present in measurements
    with pytest.raises(KeyError):
        measurements.get_mean("invalid_key")
