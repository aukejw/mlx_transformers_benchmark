import mlx.core as mx
import mlx.nn
import pytest
import torch
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import BatchEncoding, GemmaTokenizerFast
from transformers.models.gemma3.modeling_gemma3 import Gemma3PreTrainedModel

from mtb.llm_benchmarks.gemma import GemmaBenchmark


@pytest.fixture(scope="session")
def benchmark_torch():
    benchmark = GemmaBenchmark()
    benchmark.setup_torch()
    return benchmark


@pytest.fixture(scope="session")
def benchmark_mlx():
    benchmark = GemmaBenchmark()
    benchmark.setup_mlx()
    return benchmark


bfloat16_response = "Okay, here’s a story about Albert Einstein, aiming for a balance of his brilliance, his struggles"


class TestGemmaBenchmark:
    def test_setup_generate_torch(self, benchmark_torch):
        assert isinstance(benchmark_torch.model, Gemma3PreTrainedModel)
        assert isinstance(benchmark_torch.tokenizer, GemmaTokenizerFast)
        assert isinstance(benchmark_torch.model_input, BatchEncoding)
        assert isinstance(benchmark_torch.model_input["input_ids"], torch.Tensor)

        timing = benchmark_torch.run_torch_generate()
        assert isinstance(timing["generation"], str)
        assert isinstance(timing["prompt_tps"], float)
        assert isinstance(timing["generation_tps"], float)
        assert timing["generation"].startswith(bfloat16_response)

    def test_setup_generate_mlx(self, benchmark_mlx):
        assert isinstance(benchmark_mlx.model, mlx.nn.Module)
        assert isinstance(benchmark_mlx.tokenizer, TokenizerWrapper)
        assert isinstance(benchmark_mlx.model_input, BatchEncoding)
        assert isinstance(benchmark_mlx.model_input["input_ids"], mx.array)

        timing = benchmark_mlx.run_mlx_generate()
        assert isinstance(timing["generation"], str)
        assert isinstance(timing["prompt_tps"], float)
        assert isinstance(timing["generation_tps"], float)
        assert timing["generation"].startswith(bfloat16_response)
