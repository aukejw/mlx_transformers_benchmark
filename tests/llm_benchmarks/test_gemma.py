import mlx.core as mx
import mlx.nn
import pytest
import torch
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import BatchEncoding, GemmaTokenizerFast
from transformers.models.gemma3.modeling_gemma3 import Gemma3PreTrainedModel

from mtb import FLAG_ON_MAC
from mtb.llm_benchmarks.gemma import Gemma3OneBillionBenchmark


@pytest.fixture(scope="session")
def benchmark_torch():
    benchmark = Gemma3OneBillionBenchmark(max_num_tokens=30)
    benchmark.setup(framework="torch", backend="mps", dtype="bfloat16")
    benchmark.set_prompt("Write a story about Einstein", batch_size=1)
    return benchmark


@pytest.fixture(scope="session")
def benchmark_mlx():
    benchmark = Gemma3OneBillionBenchmark(max_num_tokens=30)
    benchmark.setup(framework="mlx", backend="metal", dtype="bfloat16")
    benchmark.set_prompt("Write a story about Einstein", batch_size=1)
    return benchmark


bfloat16_response = (
    "Okay, here’s a story about Albert Einstein, aiming for a balance of his brilliance, "
    "his struggles, and a touch of his quiet humanity."
)


class TestGemmaBenchmark:
    @pytest.mark.skipif(not FLAG_ON_MAC, reason="Must run on Mac")
    @pytest.mark.skipif(not torch.mps.is_available(), reason="Must run on MPS backend")
    def test_setup_generate_torch(self, benchmark_torch):
        assert isinstance(benchmark_torch.model, Gemma3PreTrainedModel)
        assert isinstance(benchmark_torch.tokenizer, GemmaTokenizerFast)
        assert isinstance(benchmark_torch.model_input, BatchEncoding)
        assert isinstance(benchmark_torch.model_input["input_ids"], torch.Tensor)

        timing = benchmark_torch.run_torch_generate()
        assert isinstance(timing["generation"], str)
        assert timing["prompt_tps"] > 0
        assert timing["prompt_time_sec"] > 0
        assert timing["generation_tps"] > 0
        assert timing["generation"].startswith(bfloat16_response)

    @pytest.mark.skipif(not FLAG_ON_MAC, reason="Must run on Mac")
    def test_setup_generate_mlx(self, benchmark_mlx):
        assert isinstance(benchmark_mlx.model, mlx.nn.Module)
        assert isinstance(benchmark_mlx.tokenizer, TokenizerWrapper)
        assert isinstance(benchmark_mlx.model_input, BatchEncoding)
        assert isinstance(benchmark_mlx.model_input["input_ids"], mx.array)

        timing = benchmark_mlx.run_mlx_generate()
        assert timing["prompt_tps"] > 0
        assert timing["prompt_time_sec"] > 0
        assert timing["generation_tps"] > 0
        assert timing["generation"].startswith(bfloat16_response)
