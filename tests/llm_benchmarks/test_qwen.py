import mlx
import mlx.core as mx
import pytest
import torch
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import BatchEncoding, Qwen2ForCausalLM, Qwen2TokenizerFast

from mtb import FLAG_ON_MAC
from mtb.llm_benchmarks.qwen import Qwen2p5_0p5B_it_Benchmark


@pytest.fixture(scope="session")
def benchmark_torch():
    torch.manual_seed(0)
    benchmark = Qwen2p5_0p5B_it_Benchmark(max_num_tokens=10)
    benchmark.setup(framework="torch", backend="mps", dtype="bfloat16")
    benchmark.set_prompt("Write a story about Einstein", batch_size=1)
    return benchmark


@pytest.fixture(scope="session")
def benchmark_mlx():
    torch.manual_seed(0)
    benchmark = Qwen2p5_0p5B_it_Benchmark(max_num_tokens=10)
    benchmark.setup(framework="mlx", backend="metal", dtype="bfloat16")
    benchmark.set_prompt("Write a story about Einstein", batch_size=1)
    return benchmark


class TestQwenBenchmark:
    @pytest.mark.skipif(not FLAG_ON_MAC, reason="Must run on Mac")
    @pytest.mark.skipif(not torch.mps.is_available(), reason="Must run on MPS backend")
    def test_setup_generate_torch(self, benchmark_torch):
        assert isinstance(benchmark_torch.model, Qwen2ForCausalLM)
        assert isinstance(benchmark_torch.tokenizer, Qwen2TokenizerFast)
        assert isinstance(benchmark_torch.model_input, BatchEncoding)
        assert isinstance(benchmark_torch.model_input["input_ids"], torch.Tensor)

        timing = benchmark_torch.run_torch_generate()
        assert timing.prompt_tps > 0
        assert timing.prompt_time_sec > 0
        assert timing.generation_tps > 0

        bfloat16_response = "Einstein was born in 1879"
        assert timing.response.startswith(bfloat16_response)

    @pytest.mark.skipif(not FLAG_ON_MAC, reason="Must run on Mac")
    def test_setup_generate_mlx(self, benchmark_mlx):
        assert isinstance(benchmark_mlx.model, mlx.nn.Module)
        assert isinstance(benchmark_mlx.tokenizer, TokenizerWrapper)
        assert isinstance(benchmark_mlx.model_input, BatchEncoding)
        assert isinstance(benchmark_mlx.model_input["input_ids"], mx.array)

        timing = benchmark_mlx.run_mlx_generate()
        assert timing.prompt_tps > 0
        assert timing.prompt_time_sec > 0
        assert timing.generation_tps > 0

        bfloat16_response = "Einstein was a brilliant and innovative mind"
        assert timing.response.startswith(bfloat16_response)
