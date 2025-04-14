import time
from typing import Any

import mlx.core as mx
import mlx.nn
import mlx_lm
import mlx_lm.tokenizer_utils
import torch
from transformers import AutoTokenizer, BatchEncoding, Gemma3ForCausalLM

from mtb.hf_utils import set_hf_home

# from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark


class BaseLLMBenchmark:
    def __init__(self, name: str):
        self.name = name

        self.input_tensor = None
        self.model = None
        self.tokenizer = None

        set_hf_home()

    def setup_torch(self):
        raise NotImplementedError

    def setup_mlx(self):
        raise NotImplementedError

    def run_torch(self):
        raise NotImplementedError

    def run_mlx(self):
        raise NotImplementedError


class GemmaBenchmark(BaseLLMBenchmark):
    dtype_to_model_id = {
        torch.bfloat16: "google/gemma-3-1b-it",
        mx.bfloat16: "mlx-community/gemma-3-1b-it-bf16",
    }

    def __init__(
        self,
    ):
        model_name = "gemma-3-4b-it"
        name = f"GemmaBencmark(model_name={model_name})"

        super().__init__(name=name)

    def setup_torch(self):
        self._dtype = torch.bfloat16
        self._device = torch.device("mps")
        torch.set_default_device(self._device)

        model_id = self.dtype_to_model_id[self._dtype]

        self.model = Gemma3ForCausalLM.from_pretrained(
            model_id,
            device_map=self._device,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model_input: BatchEncoding = self._get_model_input(tensor_type="pt")
        self.model_input.to(self._device)

    def setup_mlx(self):
        self._dtype = mx.bfloat16
        self._device = mx.Device(mx.DeviceType.gpu)
        mx.set_default_device(self._device)
        model_id = self.dtype_to_model_id[self._dtype]

        model, tokenizer = mlx_lm.load(model_id)
        self.model: mlx.nn.Module = model
        self.tokenizer: mlx_lm.tokenizer_utils.TokenizerWrapper = tokenizer

        self.model_input: mx.array = self._get_model_input(tensor_type="mlx")

    @torch.inference_mode()
    def run_torch_generate(self):
        input_ids = self.model_input["input_ids"]
        num_prompt_tokens = input_ids.shape[1]

        max_num_tokens = 100

        # Time processing of prompt, initializing kv cache
        tic = time.perf_counter()
        # TODO
        time.sleep(0.1)

        prompt_seconds = time.perf_counter() - tic
        prompt_tps = num_prompt_tokens / prompt_seconds

        # Generate tokens
        tic = time.perf_counter()
        generation = self.model.generate(
            **self.model_input,
            max_new_tokens=max_num_tokens,
            do_sample=False,
            top_k=None,
            top_p=None,
        )
        generation = self.tokenizer.batch_decode(generation[0, num_prompt_tokens:])
        generation = "".join(generation)

        generation_seconds = time.perf_counter() - tic
        generation_tps = max_num_tokens / generation_seconds

        return dict(
            generation=generation,
            prompt_tps=prompt_tps,
            generation_tps=generation_tps,
        )

    def run_mlx_generate(self):
        # use stream_generate instead of generate, its response is more useful
        generation = ""
        for response in mlx_lm.stream_generate(
            self.model,
            self.tokenizer,
            max_tokens=100,
            prompt=self.model_input["input_ids"][0],
        ):
            generation += response.text

        return dict(
            generation=generation,
            prompt_tps=response.prompt_tps,
            generation_tps=response.generation_tps,
            peak_memory=response.peak_memory,
        )

    def _get_model_input(self, tensor_type: str) -> Any:
        # Input-finetuned models ('-pi' suffix) expect a specific format
        prompt_string = "Write a story about Einstein"
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_string}],
            },
        ]

        model_input = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=tensor_type,
        )

        return model_input
