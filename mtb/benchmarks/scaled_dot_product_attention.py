from typing import Optional, Tuple

import mlx.core as mx
import torch

from mtb.attention_mask import (
    create_mlx_attention_mask,
    create_torch_attention_mask,
    validate_attention_kwargs,
)
from mtb.benchmarks.base_benchmark import BaseBenchmark


class ScaledDotProductAttentionBenchmark(BaseBenchmark):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_heads: int = 8,
        mask_type: Optional[str] = None,
    ):
        num_features = input_shape[2]
        name = (
            f"ScaledDotProductAttention("
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
        self.head_dim = num_features // num_heads
        self.scale = 1 / self.head_dim**0.5

    def setup_torch(self):
        batch_size, num_tokens, num_features = self.input_shape

        self.input_tensor = (
            self.input_tensor.reshape(
                batch_size,
                num_tokens,
                self.num_heads,
                num_features // self.num_heads,
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        self.torch_function = torch.nn.functional.scaled_dot_product_attention

        self.mask = create_torch_attention_mask(
            mask_type=self.mask_type,
            num_tokens=num_tokens,
            device=self._device,
            dtype=self._dtype,
        )

    def setup_mlx(self):
        batch_size, num_tokens, num_features = self.input_shape

        self.input_tensor = mx.contiguous(
            self.input_tensor.reshape(
                batch_size,
                num_tokens,
                self.num_heads,
                num_features // self.num_heads,
            ).transpose(0, 2, 1, 3)
        )

        self.mask = create_mlx_attention_mask(
            mask_type=self.mask_type,
            num_tokens=num_tokens,
            device=self._device,
            dtype=self._dtype,
            compile=self._compile,
        )

        self.mlx_function = mx.fast.scaled_dot_product_attention
        if self._compile:
            self.mlx_function = mx.compile(self.mlx_function)

    @torch.inference_mode()
    def run_torch(self) -> torch.Tensor:
        q = k = v = self.input_tensor
        fn = self.torch_function
        y = fn(q, k, v, scale=self.scale, attn_mask=self.mask)
        return y

    def run_mlx(self) -> mx.array:
        q = k = v = self.input_tensor
        fn = self.mlx_function
        y = fn(q, k, v, scale=self.scale, mask=self.mask)
        return y
