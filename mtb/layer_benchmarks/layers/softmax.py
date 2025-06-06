import mlx.core as mx
import torch

from mtb.layer_benchmarks.base_layer_benchmark import BaseLayerBenchmark


class SoftmaxBenchmark(BaseLayerBenchmark):
    def __init__(
        self,
        feature_dim: int,
    ):
        name = f"Softmax(dim={feature_dim})"
        super().__init__(name=name, feature_dim=feature_dim)

    def setup_torch(self):
        self.torch_function = torch.nn.functional.softmax

    def setup_mlx(self):
        self.mlx_function = mx.softmax
        if self._compile:
            self.mlx_function = mx.compile(self.mlx_function)

    @torch.inference_mode()
    def run_torch(self) -> torch.Tensor:
        x = self.input_tensor
        fn = self.torch_function
        y = fn(x, dim=2)
        return y

    def run_mlx(self) -> mx.array:
        x = self.input_tensor
        fn = self.mlx_function
        y = fn(x, axis=2)
        return y
