import numpy as np
import pytest

from mtb.measurement import Measurement, Measurements


def test_measurement():
    measurements = [0.1, 0.2, 0.5]

    measurement = Measurement(measurements)

    assert measurement.mean == np.mean(measurements)
    assert measurement.std == np.std(measurements)
    assert measurement.median == np.median(measurements)

    assert str(measurement) == (
        "Measurement("
        "\n  num_measurements=3, "
        "\n  median=0.2000, "
        "\n  mean=0.2667, "
        "\n  std=0.1700, "
        "\n)"
    )


def test_measurements():
    measurements = Measurements()
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
