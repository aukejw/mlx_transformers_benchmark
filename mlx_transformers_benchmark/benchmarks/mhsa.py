from typing import List

import mlx
import mlx.core as mx
import mlx.nn
import torch
import torch.nn

from mlx_transformers_benchmark.benchmarks.base_benchmark import BaseBenchmark


class MhsaBenchmark(BaseBenchmark):
    """Benchmark LayerNorm implementations."""

    def __init__(self, input_shapes: List, num_heads: int = 8):
        super().__init__(
            name=f"MHSA(dim={input_shapes[0][2]}, num_heads={num_heads})",
            input_shapes=input_shapes,
        )
        self.num_heads = num_heads

    def _setup_torch(self, backend: str, dtype: str):
        batch_size, num_tokens, num_features = self.input_shapes[0]

        self.torch_function = torch.nn.MultiheadAttention(
            embed_dim=num_features,
            num_heads=self.num_heads,
            bias=True,
            batch_first=True,  # mlx only has batch_first
            device=torch.device(backend),
            dtype=dtype,
        )
        self.torch_function.eval()

    def _setup_mlx(self, backend: str, dtype: str, compile: bool):
        batch_size, num_tokens, num_features = self.input_shapes[0]

        self.mlx_function = mlx.nn.MultiHeadAttention(
            dims=num_features,
            num_heads=self.num_heads,
            bias=True,
        )
        self.mlx_function.eval()

        if compile:
            self.mlx_function = mx.compile(self.mlx_function)

    @torch.inference_mode()
    def _run_torch(self, backend: str) -> torch.Tensor:
        x = self.input_tensors[0]
        fn = self.torch_function
        y = fn(x, x, x)
        return y

    def _run_mlx(self, backend: str) -> mx.array:
        x = self.input_tensors[0]
        fn = self.mlx_function
        y = fn(x, x, x)
        return y
