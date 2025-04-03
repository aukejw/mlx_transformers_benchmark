from typing import List

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
