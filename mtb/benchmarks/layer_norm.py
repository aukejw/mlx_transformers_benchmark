from typing import Tuple

import mlx
import mlx.core as mx
import mlx.nn
import torch
import torch.nn

from mtb.benchmarks.base_benchmark import BaseBenchmark


class LayerNormBenchmark(BaseBenchmark):
    def __init__(self, input_shape: Tuple[int, int, int]):
        super().__init__(
            name=f"LayerNorm(dim={input_shape[2]})",
            input_shape=input_shape,
        )

    def setup_torch(self):
        batch_size, num_tokens, num_features = self.input_shape

        self.torch_function = torch.nn.LayerNorm(
            normalized_shape=num_features,
            elementwise_affine=True,
            bias=True,
            device=self._backend,
        )

    def setup_mlx(self):
        batch_size, num_tokens, num_features = self.input_shape

        self.mlx_function = mlx.nn.LayerNorm(
            dims=num_features,
            affine=True,
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
