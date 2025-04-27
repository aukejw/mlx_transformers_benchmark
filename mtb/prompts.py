import random

from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark


def find_prompt_for_llm_benchmark(
    num_tokens: int,
    benchmark: BaseLLMBenchmark,
):
    """Find a prompt of the desired number of tokens.

    As each benchmark tokenizes and formats prompts differently, we need to find
    prompt that has the right number of tokens, but that still makes sense for the
    model.

    Args:
        num_tokens: The number of tokens to find a prompt for.
        benchmark: The benchmark, which contains a tokenization function.

    Returns:
        A prompt with the correct number of tokens for the tokenizer and message template.

    """
    # initial guess: we need 1 character per token
    text_length = int(num_tokens)
    num_prompt_tokens = None

    while num_prompt_tokens != num_tokens:
        prompt = get_random_prompt(text_length=text_length)
        prompt_tokens = benchmark.format_prompt(prompt)
        num_prompt_tokens = len(prompt_tokens[0])

        if num_prompt_tokens < num_tokens:
            # prompt too short, increase length by at least 1 character
            text_length += max(1, int((num_tokens - num_prompt_tokens) / 2))
        elif num_prompt_tokens > num_tokens:
            # prompt too long, decrease length by at least 1 character
            text_length -= max(1, int((num_prompt_tokens - num_tokens) / 2))

    return prompt


def get_random_prompt(text_length: int) -> str:
    """Get a prompt of the specified length by generating some text.

    This randomizes initial tokens to avoid prompt caching. Note that the
    text_length is in characters, not tokens.

    We prefer if the prompt is an actual task, instead of just any random
    tokens. This is so we can check the model correctness, and hopefully
    avoid early stopping.

    As an example, the prompt could look like this:

        0, 3, 2. Write a story about Einstein

    """
    task_string = "Write a story about Einstein"

    if text_length < len(task_string) + 2:
        raise ValueError(
            f"Text length {text_length} is too short for task string '{task_string}'"
        )

    # add random prefix sequence
    prefix_length = text_length - len(task_string) - 2
    prompt = "".join(str(random.randint(0, 10)) for _ in range(prefix_length)) + ". "

    # actual task
    prompt += task_string

    return prompt
