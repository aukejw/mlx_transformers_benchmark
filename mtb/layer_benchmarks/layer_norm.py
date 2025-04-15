import mlx
import mlx.core as mx
import mlx.nn
import torch
import torch.nn

from mtb.layer_benchmarks.base_layer_benchmark import BaseLayerBenchmark


class LayerNormBenchmark(BaseLayerBenchmark):
    def __init__(
        self,
        feature_dim: int,
    ):
        super().__init__(name=f"LayerNorm(dim={feature_dim})", feature_dim=feature_dim)

    def setup_torch(self):
        self.torch_function = torch.nn.LayerNorm(
            normalized_shape=self.feature_dim,
            elementwise_affine=True,
            bias=True,
            device=self._backend,
        )

    def setup_mlx(self):
        self.mlx_function = mlx.nn.LayerNorm(
            dims=self.feature_dim,
            affine=True,
            bias=True,
        )
        self.mlx_function.set_dtype(self._dtype)

        if self._compile:
            self.mlx_function = mx.compile(self.mlx_function)

    @torch.inference_mode()
    def run_torch(self) -> torch.Tensor:
        x = self.input_tensor
        fn = self.torch_function
        y = fn(x)
        return y

    def run_mlx(self) -> mx.array:
        x = self.input_tensor
        fn = self.mlx_function
        y = fn(x)
        return y
