import numpy as np

from mtb.measurement import Measurement


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
