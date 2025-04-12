from typing import Tuple

import mlx
import mlx.core as mx
import mlx.nn
import torch
import torch.nn

from mtb.benchmarks.base_benchmark import BaseBenchmark


class MhsaBenchmark(BaseBenchmark):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_heads: int = 8,
    ):
        num_features = input_shape[2]
        name = f"MHSA(dim={num_features}, num_heads={num_heads})"

        super().__init__(
            name=name,
            input_shape=input_shape,
        )
        self.num_heads = num_heads

        if num_features % self.num_heads != 0:
            raise ValueError(
                f"num_features={num_features} must be divisible by num_heads={self.num_heads}"
            )

    def setup_torch(self):
        batch_size, num_tokens, num_features = self.input_shape

        self.torch_function = torch.nn.MultiheadAttention(
            embed_dim=num_features,
            num_heads=self.num_heads,
            bias=True,
            batch_first=True,  # mlx only has batch_first
            device=self._device,
            dtype=self._dtype,
        )
        self.torch_function.eval()

    def setup_mlx(self):
        batch_size, num_tokens, num_features = self.input_shape

        self.mlx_function = mlx.nn.MultiHeadAttention(
            dims=num_features,
            num_heads=self.num_heads,
            bias=True,
        )
        self.mlx_function.eval()
        self.mlx_function.set_dtype(self._dtype)

        if self._compile:
            self.mlx_function = mx.compile(self.mlx_function)

    @torch.inference_mode()
    def run_torch(self) -> torch.Tensor:
        q = k = v = self.input_tensor
        fn = self.torch_function
        y = fn(q, k, v)
        return y

    def run_mlx(self) -> mx.array:
        q = k = v = self.input_tensor
        fn = self.mlx_function
        y = fn(q, k, v)
        return y
