from typing import Optional

import mlx
import mlx.core as mx
import mlx.nn
import torch
import torch.nn

from mtb.attention_mask import (
    create_mlx_attention_mask,
    create_torch_attention_mask,
    validate_attention_kwargs,
)
from mtb.layer_benchmarks.base_layer_benchmark import BaseLayerBenchmark


class MhsaBenchmark(BaseLayerBenchmark):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 8,
        mask_type: Optional[str] = None,
    ):
        name = (
            f"MHSA("
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
        self.num_heads = num_heads
        self.mask_type = mask_type

        # placeholder variables
        self.mask = None

    def set_input_tensor(
        self,
        batch_size: int,
        sequence_length: int,
    ):
        super().set_input_tensor(batch_size, sequence_length)

        if self._framework == "torch":
            self.mask = create_torch_attention_mask(
                mask_type=self.mask_type,
                num_tokens=sequence_length,
                device=self._device,
                dtype=self._dtype,
                compile=False,
            )
        elif self._framework == "mlx":
            self.mask = create_mlx_attention_mask(
                mask_type=self.mask_type,
                num_tokens=sequence_length,
                device=self._device,
                dtype=self._dtype,
                compile=self._compile,
            )
        else:
            raise NotImplementedError("Framework '{self._framework}' not supported")

    def setup_torch(self):
        self.torch_function = torch.nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=self.num_heads,
            bias=True,
            batch_first=True,  # mlx only has batch_first
            device=self._device,
            dtype=self._dtype,
        )
        self.torch_function.eval()

    def setup_mlx(self):
        self.mlx_function = mlx.nn.MultiHeadAttention(
            dims=self.feature_dim,
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
