from typing import Tuple

import mlx
import mlx.core as mx
import mlx.nn
import torch
import torch.nn

from mtb.benchmarks.base_benchmark import BaseBenchmark


class LinearBenchmark(BaseBenchmark):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
    ):
        num_features = input_shape[2]
        name = f"Linear(in={num_features}, out={num_features})"

        super().__init__(
            name=name,
            input_shape=input_shape,
        )

    def setup_torch(self):
        batch_size, num_tokens, num_features = self.input_shape

        self.torch_function = torch.nn.Linear(
            in_features=num_features,
            out_features=num_features,
            bias=True,
            device=self._device,
            dtype=self._dtype,
        )

    def setup_mlx(self):
        batch_size, num_tokens, num_features = self.input_shape

        self.mlx_function = mlx.nn.Linear(
            input_dims=num_features,
            output_dims=num_features,
            bias=True,
        )
        self.mlx_function.set_dtype(self._dtype)

        if self._compile:
            self.mlx_function = mx.compile(self.mlx_function)

    @torch.inference_mode()
    def run_torch(self) -> torch.Tensor:
        x = self.input_tensor
        fn = self.torch_function
        y = fn(x)
        return y

    def run_mlx(self) -> mx.array:
        x = self.input_tensor
        fn = self.mlx_function
        y = fn(x)
        return y
