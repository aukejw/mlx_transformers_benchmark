from typing import Tuple

import mlx
import mlx.core as mx
import mlx.nn
import torch
import torch.nn

from mtb.attention_mask import create_mlx_attention_mask, create_torch_attention_mask
from mtb.benchmarks.base_benchmark import BaseBenchmark


class TransformerDecoderLayerBenchmark(BaseBenchmark):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_heads: int = 8,
        dropout: float = 0.1,
        norm_first: bool = True,
        mask_type: str = None,
    ):
        num_features = input_shape[2]

        name = (
            f"TransformerDecoderLayer("
            f"dim={num_features}, "
            f"num_heads={num_heads}, "
            f"mask={mask_type})"
        )
        super().__init__(
            name=name,
            input_shape=input_shape,
        )
        assert num_features % num_heads == 0, (num_features, num_heads)
        assert dropout >= 0.0 and dropout <= 1.0, dropout
        assert mask_type in (None, "causal"), mask_type

        self.num_heads = num_heads
        self.dropout = dropout
        self.norm_first = norm_first
        self.mask_type = mask_type

        # placeholder variables
        self.mask = None
        self.memory = None
        self.memory_mask = None

    def _setup_torch(self, backend: str, dtype: str):
        batch_size, num_tokens, num_features = self.input_shape

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

        self.mask = create_torch_attention_mask(
            mask_type=self.mask_type,
            attention_layer=self.torch_function.self_attn,
            num_tokens=num_tokens,
        )
        self.memory = torch.randn(
            batch_size,
            num_tokens,
            num_features,
            device=backend,
            dtype=dtype,
        )
        self.mask = create_torch_attention_mask(
            mask_type=None,
            attention_layer=self.torch_function.multihead_attn,
            num_tokens=num_tokens,
            compile=False,
        )

    def _setup_mlx(self, backend: str, dtype: str, compile: bool):
        batch_size, num_tokens, num_features = self.input_shape

        self.mlx_function = mlx.nn.TransformerDecoderLayer(
            dims=num_features,
            mlp_dims=4 * num_features,
            num_heads=self.num_heads,
            dropout=self.dropout,
            norm_first=self.norm_first,
        )
        self.mlx_function.eval()

        self.mask = create_mlx_attention_mask(
            mask_type=self.mask_type,
            attention_layer=self.mlx_function.self_attention,
            num_tokens=num_tokens,
            compile=compile,
        )
        self.memory = mx.random.normal(
            (batch_size, num_tokens, num_features),
            dtype=dtype,
        )
        self.memory_mask = create_mlx_attention_mask(
            mask_type=None,
            attention_layer=self.mlx_function.cross_attention,
            num_tokens=num_tokens,
            compile=compile,
        )

        if compile:
            self.mlx_function = mx.compile(self.mlx_function)

    @torch.inference_mode()
    def _run_torch(self, backend: str) -> torch.Tensor:
        x = self.input_tensors[0]
        fn = self.torch_function
        y = fn(
            tgt=x,
            tgt_mask=self.mask,
            memory=self.memory,
            memory_mask=self.memory_mask,
        )
        return y

    def _run_mlx(self, backend: str) -> mx.array:
        x = self.input_tensors[0]
        fn = self.mlx_function
        y = fn(
            x=x,
            x_mask=self.mask,
            memory=self.memory,
            memory_mask=self.memory_mask,
        )
        return y

    def teardown(self, framework: str, backend: str):
        del self.mask
        del self.memory

        super().teardown(framework, backend)
