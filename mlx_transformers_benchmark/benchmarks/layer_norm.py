from typing import List, Optional

import mlx
import mlx.core as mx
import mlx.nn
import torch
import torch.nn

from mlx_transformers_benchmark.benchmarks.base_benchmark import BaseBenchmark


class LayerNormBenchmark(BaseBenchmark):
    def __init__(self, input_shapes: List):
        super().__init__(
            name=f"LayerNorm(dim={input_shapes[0][2]})",
            input_shapes=input_shapes,
        )

    def _setup_torch(self, backend: str, dtype: str):
        batch_size, num_tokens, num_features = self.input_shapes[0]

        self.torch_function = torch.nn.LayerNorm(
            normalized_shape=num_features,
            elementwise_affine=True,
            bias=True,
            device=backend,
        )

    def _setup_mlx(self, backend: str, dtype: str, compile: bool):
        batch_size, num_tokens, num_features = self.input_shapes[0]

        self.mlx_function = mlx.nn.LayerNorm(
            dims=num_features,
            affine=True,
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
