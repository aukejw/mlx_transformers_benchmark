from typing import List

import mlx
import mlx.core as mx
import mlx.nn
import torch
import torch.nn

from mlx_transformers_benchmark.benchmarks.base_benchmark import BaseBenchmark


class TransformerDecoderLayer(BaseBenchmark):
    def __init__(
        self,
        input_shapes: List,
        num_heads: int = 8,
        dropout: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__(
            name=f"TransformerDecoderLayer(dim={input_shapes[0][2]})",
            input_shapes=input_shapes,
        )
        self.num_heads = num_heads
        self.dropout = dropout
        self.norm_first = norm_first

    def _setup_torch(self, backend: str, dtype: str):
        batch_size, num_tokens, num_features = self.input_shapes[0]

        self.torch_function = torch.nn.TransformerDecoderLayer(
            d_model=num_features,
            dim_feedforward=num_features * 4,
            nhead=self.num_heads,
            dropout=self.dropout,
            norm_first=self.norm_first,
            batch_first=True,
            bias=True,  # mlx has bias True by default
            device=backend,
            dtype=dtype,
        )
        self.torch_function.eval()

        self.mask = None

    def _setup_mlx(self, backend: str, dtype: str, compile: bool):
        batch_size, num_tokens, num_features = self.input_shapes[0]

        self.mlx_function = mlx.nn.TransformerDecoderLayer(
            dims=num_features,
            mlp_dims=4 * num_features,
            num_heads=self.num_heads,
            dropout=self.dropout,
            norm_first=self.norm_first,
        )
        self.mlx_function.eval()

        self.mask = None
        if compile:
            # compiled functions cannot handle mask=None inputs
            self.mask = mx.ones(num_tokens, num_tokens)
            self.mlx_function = mx.compile(self.mlx_function)

    @torch.inference_mode()
    def _run_torch(self, backend: str) -> torch.Tensor:
        x = self.input_tensors[0]
        fn = self.torch_function
        y = fn(x, src_mask=self.mask)
        return y

    def _run_mlx(self, backend: str) -> mx.array:
        x = self.input_tensors[0]
        fn = self.mlx_function
        y = fn(x, mask=self.mask)
        return y
