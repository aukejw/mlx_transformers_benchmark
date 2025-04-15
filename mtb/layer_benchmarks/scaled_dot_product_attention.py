from typing import Optional

import mlx.core as mx
import torch

from mtb.attention_mask import create_attention_mask, validate_attention_kwargs
from mtb.layer_benchmarks.base_layer_benchmark import BaseLayerBenchmark


class ScaledDotProductAttentionBenchmark(BaseLayerBenchmark):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 8,
        mask_type: Optional[str] = None,
    ):
        name = (
            f"ScaledDotProductAttention("
            f"dim={feature_dim}, "
            f"num_heads={num_heads}, "
            f"mask={mask_type})"
        )
        super().__init__(name=name)

        validate_attention_kwargs(
            feature_dim=feature_dim,
            num_heads=num_heads,
            mask_type=mask_type,
        )

        self.num_heads = num_heads
        self.mask_type = mask_type
        self.head_dim = feature_dim // num_heads
        self.scale = 1 / self.head_dim**0.5

    def set_input_tensor(self, batch_size: int, sequence_length: int):
        self._batch_size = batch_size
        self._sequence_length = sequence_length

        input_shape = (
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        if self._framework == "torch":
            self.input_tensor = torch.rand(
                input_shape, device=self._device, dtype=self._dtype
            )
        elif self._framework == "mlx":
            self.input_tensor = mx.random.normal(
                input_shape,
            ).astype(self._dtype)
        else:
            raise NotImplementedError(f"Unknown framework {self._framework}")

        self.mask = create_attention_mask(
            framework=self._framework,
            mask_type=self.mask_type,
            num_tokens=sequence_length,
            device=self._device,
            dtype=self._dtype,
            compile=self._compile,
        )

    def setup_torch(self):
        self.torch_function = torch.nn.functional.scaled_dot_product_attention

    def setup_mlx(self):
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

    def teardown(self):
        del self.mask
        self.mask = None
        super().teardown()
