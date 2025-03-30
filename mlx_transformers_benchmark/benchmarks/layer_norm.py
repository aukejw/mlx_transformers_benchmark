import torch
from typing import List, Optional
import torch.nn
import mlx.nn
import mlx
import mlx.core as mx

from mlx_transformers_benchmark.benchmarks.base_benchmark import BaseBenchmark


class LayerNormBenchmark(BaseBenchmark):
    def __init__(self, input_shapes: List):
        super().__init__(
            name="LayerNorm",
            input_shapes=input_shapes,
        )

    def _setup_torch(self, device: str, dtype: str):
        self.torch_function = torch.nn.LayerNorm(
            normalized_shape=self.input_shapes[0][1:],
            elementwise_affine=True,
            bias=True,
            device=device,
            dtype=dtype,
        )

    def _setup_mlx(self, device: str, dtype: str):
        self.mlx_function = mlx.nn.LayerNorm(
            dims=self.input_shapes[0][1],
            affine=True,
            bias=True,
        )
        self.compiled_mlx_function = mx.compile(self.mlx_function)

    def _run_torch(self, device: str) -> torch.Tensor:
        """Run the benchmark using torch."""
        x = self.input_tensors_torch[0]
        fn = self.torch_function
        y = fn(x)

    def _run_mlx(self, deviec: str, compile: bool) -> mlx.array:
        """Run the benchmark using mlx."""
        x = self.input_tensors_mlx[0]
        fn = self.compiled_mlx_function if compile else self.mlx_function
        y = fn(x)
        mlx.eval(y)
