import mlx
import mlx.core as mx
import mlx.nn
import torch
import torch.nn

from mtb.layer_benchmarks.base_layer_benchmark import BaseLayerBenchmark


class LinearBenchmark(BaseLayerBenchmark):
    def __init__(
        self,
        feature_dim: int,
    ):
        name = f"Linear(in={feature_dim}, out={feature_dim})"

        super().__init__(
            name=name,
            feature_dim=feature_dim,
        )

    def setup_torch(self):
        self.torch_function = torch.nn.Linear(
            in_features=self.feature_dim,
            out_features=self.feature_dim,
            bias=True,
            device=self._device,
            dtype=self._dtype,
        )

    def setup_mlx(self):
        self.mlx_function = mlx.nn.Linear(
            input_dims=self.feature_dim,
            output_dims=self.feature_dim,
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
