from typing import Tuple

import mlx
import mlx.core as mx
import mlx.nn
import torch
import torch.nn

from mtb.attention_mask import create_mlx_attention_mask, create_torch_attention_mask
from mtb.benchmarks.base_benchmark import BaseBenchmark


class TransformerEncoderLayerBenchmark(BaseBenchmark):
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
            f"TransformerEncoderLayer("
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

    def setup_torch(self):
        batch_size, num_tokens, num_features = self.input_shape

        self.torch_function = torch.nn.TransformerEncoderLayer(
            d_model=num_features,
            dim_feedforward=num_features * 4,
            nhead=self.num_heads,
            dropout=self.dropout,
            norm_first=self.norm_first,
            batch_first=True,
            bias=True,  # mlx has bias True by default
            device=self._device,
            dtype=self._dtype,
        )
        self.torch_function.eval()

        self.mask = create_torch_attention_mask(
            mask_type=self.mask_type,
            attention_layer=self.torch_function.self_attn,
            num_tokens=num_tokens,
        )

    def setup_mlx(self):
        batch_size, num_tokens, num_features = self.input_shape

        self.mlx_function = mlx.nn.TransformerEncoderLayer(
            dims=num_features,
            mlp_dims=4 * num_features,
            num_heads=self.num_heads,
            dropout=self.dropout,
            norm_first=self.norm_first,
        )
        self.mlx_function.eval()
        self.mlx_function.set_dtype(self._dtype)

        self.mask = create_mlx_attention_mask(
            mask_type=self.mask_type,
            attention_layer=self.mlx_function.attention,
            num_tokens=num_tokens,
            compile=self._compile,
        )
        if self._compile:
            self.mlx_function = mx.compile(self.mlx_function)

    @torch.inference_mode()
    def run_torch(self) -> torch.Tensor:
        x = self.input_tensor
        fn = self.torch_function
        y = fn(x, src_mask=self.mask)
        return y

    def run_mlx(self) -> mx.array:
        x = self.input_tensor
        fn = self.mlx_function
        y = fn(x, mask=self.mask)
        return y
