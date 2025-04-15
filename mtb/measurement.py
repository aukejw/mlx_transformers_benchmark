from typing import Dict, List

import numpy as np


class Measurement:
    def __init__(self, measurements: List[float]):
        self.measurements = np.array(measurements)

        self.num_measurements = len(self.measurements)
        self.median = np.median(self.measurements)
        self.mean = np.mean(self.measurements)
        self.std = np.std(self.measurements)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"\n  num_measurements={self.num_measurements}, "
            f"\n  median={self.median:.4f}, "
            f"\n  mean={self.mean:.4f}, "
            f"\n  std={self.std:.4f}, "
            f"\n)"
        )


class Measurements:
    """Container class for measurements."""

    def __init__(self):
        self._measurements = None
        self._num_measurements = 0

    def add(self, measurement: Dict):
        # initialize container if needed
        if self._measurements is None:
            self._measurements = dict()
            for key in measurement:
                self._measurements[key] = []

        # add values to the container
        for key in self._measurements:
            if key not in measurement:
                raise KeyError(f"Key {key} not found in measurements")

            self._measurements[key].append(measurement[key])

        self._num_measurements += 1
        return

    def get_mean(self, key: str) -> float:
        if key in self._measurements:
            return np.mean(self._measurements[key])
        else:
            raise KeyError(
                f"Key {key} not found in measurements, must be one of {list(self._measurements.keys())}"
            )

    def get_means(self) -> Dict[str, float]:
        return {key: self.get_mean(key) for key in self._measurements}

    def reset(self):
        self._measurements = None
        self._num_measurements = 0

    def __repr__(self) -> str:
        tostring = (
            f"{self.__class__.__name__}("
            f"\n  num_measurements={self._num_measurements},"
        )
        for key in self._measurements:
            tostring += f"\n  {key}={self.get_mean(key):.4f},"
        tostring += "\n)"
        return tostring
