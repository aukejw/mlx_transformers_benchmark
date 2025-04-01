from typing import List

import mlx
import mlx.core as mx
import mlx.nn
import torch
import torch.nn

from mlx_transformers_benchmark.benchmarks.base_benchmark import BaseBenchmark


class LinearBenchmark(BaseBenchmark):
    def __init__(self, input_shapes: List):
        super().__init__(
            name=f"Linear(in={input_shapes[0][2]}, out={input_shapes[0][2]})",
            input_shapes=input_shapes,
        )

    def _setup_torch(self, backend: str, dtype: str):
        batch_size, num_tokens, num_features = self.input_shapes[0]

        self.torch_function = torch.nn.Linear(
            in_features=num_features,
            out_features=num_features,
            bias=True,
            device=backend,
            dtype=dtype,
        )

    def _setup_mlx(self, backend: str, dtype: str, compile: bool):
        batch_size, num_tokens, num_features = self.input_shapes[0]

        self.mlx_function = mlx.nn.Linear(
            input_dims=num_features,
            output_dims=num_features,
            bias=True,
        )
        if compile:
            self.mlx_function = mx.compile(self.mlx_function)

    @torch.inference_mode()
    def _run_torch(self, backend: str) -> torch.Tensor:
        x = self.input_tensors[0]
        fn = self.torch_function
        y = fn(x)
        return y

    def _run_mlx(self, backend: str) -> mx.array:
        x = self.input_tensors[0]
        fn = self.mlx_function
        y = fn(x)
        return y
