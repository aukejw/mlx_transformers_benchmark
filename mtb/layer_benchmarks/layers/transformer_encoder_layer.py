from typing import Optional

import mlx
import mlx.core as mx
import mlx.nn
import torch
import torch.nn

from mtb.attention_mask import create_attention_mask, validate_attention_kwargs
from mtb.layer_benchmarks.base_layer_benchmark import BaseLayerBenchmark


class TransformerEncoderLayerBenchmark(BaseLayerBenchmark):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        norm_first: bool = True,
        mask_type: Optional[str] = None,
    ):
        name = (
            f"TransformerEncoderLayer("
            f"dim={feature_dim}, "
            f"num_heads={num_heads}, "
            f"mask={mask_type})"
        )
        super().__init__(name=name, feature_dim=feature_dim)

        validate_attention_kwargs(
            feature_dim=feature_dim,
            num_heads=num_heads,
            mask_type=mask_type,
        )
        assert dropout >= 0.0 and dropout <= 1.0, dropout

        self.num_heads = num_heads
        self.dropout = dropout
        self.norm_first = norm_first
        self.mask_type = mask_type

        # placeholder variables
        self.mask = None

    def set_input_tensor(self, batch_size, sequence_length):
        self.mask = create_attention_mask(
            framework=self._framework,
            mask_type=self.mask_type,
            num_tokens=sequence_length,
            device=self._device,
            dtype=self._dtype,
            compile=self._compile,
        )
        return super().set_input_tensor(batch_size, sequence_length)

    def setup_torch(self):
        self.torch_function = torch.nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            dim_feedforward=self.feature_dim * 4,
            nhead=self.num_heads,
            dropout=self.dropout,
            norm_first=self.norm_first,
            batch_first=True,
            bias=True,  # mlx has bias True by default
            device=self._device,
            dtype=self._dtype,
        )
        self.torch_function.eval()

    def setup_mlx(self):
        self.mlx_function = mlx.nn.TransformerEncoderLayer(
            dims=self.feature_dim,
            mlp_dims=4 * self.feature_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            norm_first=self.norm_first,
        )
        self.mlx_function.eval()
        self.mlx_function.set_dtype(self._dtype)

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

    def teardown(self):
        del self.mask
        self.mask = None
        super().teardown()
