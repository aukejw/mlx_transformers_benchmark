from typing import Optional, Tuple

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
from mtb.benchmarks.base_benchmark import BaseBenchmark


class MhsaBenchmark(BaseBenchmark):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_heads: int = 8,
        mask_type: Optional[str] = None,
    ):
        num_features = input_shape[2]
        name = (
            f"MHSA("
            f"dim={num_features}, "
            f"num_heads={num_heads}, "
            f"mask={mask_type})"
        )

        super().__init__(
            name=name,
            input_shape=input_shape,
        )
        validate_attention_kwargs(
            num_features=num_features,
            num_heads=num_heads,
            mask_type=mask_type,
        )

        self.num_heads = num_heads
        self.mask_type = mask_type

        # placeholder variables
        self.mask = None

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

        self.mask = create_torch_attention_mask(
            mask_type=self.mask_type,
            num_tokens=num_tokens,
            device=self._device,
            dtype=self._dtype,
            compile=False,
        )

    def setup_mlx(self):
        batch_size, num_tokens, num_features = self.input_shape

        self.mlx_function = mlx.nn.MultiHeadAttention(
            dims=num_features,
            num_heads=self.num_heads,
            bias=True,
        )
        self.mlx_function.eval()
        self.mlx_function.set_dtype(self._dtype)

        self.mask = create_mlx_attention_mask(
            mask_type=self.mask_type,
            num_tokens=num_tokens,
            device=self._device,
            dtype=self._dtype,
            compile=self._compile,
        )

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
