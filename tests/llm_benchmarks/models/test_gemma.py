import mlx.core as mx
import mlx.nn
import pytest
import torch
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import GemmaTokenizerFast
from transformers.models.gemma3.modeling_gemma3 import Gemma3PreTrainedModel

from mtb import FLAG_ON_MAC
from mtb.llm_benchmarks.models.gemma import Gemma3_1B_it
from mtb.run_llm_benchmark import create_benchmark


@pytest.fixture(scope="session")
def benchmark_torch():
    benchmark = create_benchmark(
        model_spec=Gemma3_1B_it,
        framework="torch",
        backend="mps",
        dtype="bfloat16",
        max_num_tokens=30,
    )
    benchmark.setup()
    return benchmark


@pytest.fixture(scope="session")
def benchmark_mlx():
    benchmark = create_benchmark(
        model_spec=Gemma3_1B_it,
        framework="mlx",
        backend="metal",
        dtype="bfloat16",
        max_num_tokens=30,
    )
    benchmark.setup()
    return benchmark


bfloat16_response = (
    "Okay, here’s a story about Albert Einstein, aiming for a balance of his brilliance, "
    "his struggles, and a touch of his quiet humanity."
)


class TestGemma:
    @pytest.mark.skipif(not FLAG_ON_MAC, reason="Must run on Mac")
    @pytest.mark.skipif(not torch.mps.is_available(), reason="Must run on MPS backend")
    def test_setup_generate_torch(self, benchmark_torch):
        assert isinstance(benchmark_torch.model, Gemma3PreTrainedModel)
        assert isinstance(benchmark_torch.tokenizer, GemmaTokenizerFast)

        prompt_tokens = benchmark_torch.format_prompt("OK")
        assert isinstance(prompt_tokens, torch.Tensor)

        timing = benchmark_torch.run_once(prompt="Write a story about Einstein")
        assert timing.prompt_tps > 0
        assert timing.prompt_time_sec > 0
        assert timing.generation_tps > 0
        assert timing.response.startswith(bfloat16_response)

    @pytest.mark.skipif(not FLAG_ON_MAC, reason="Must run on Mac")
    def test_setup_generate_mlx(self, benchmark_mlx):
        assert isinstance(benchmark_mlx.model, mlx.nn.Module)
        assert isinstance(benchmark_mlx.tokenizer, TokenizerWrapper)

        prompt_tokens = benchmark_mlx.format_prompt("OK")
        assert isinstance(prompt_tokens, mx.array)

        timing = benchmark_mlx.run_once(prompt="Write a story about Einstein")
        assert timing.prompt_tps > 0
        assert timing.prompt_time_sec > 0
        assert timing.generation_tps > 0
        assert timing.response.startswith(bfloat16_response)
