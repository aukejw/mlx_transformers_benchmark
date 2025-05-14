import ollama
from tqdm import tqdm


def try_pull_ollama_model(model_id: str):
    """Pull a model, if not already available."""
    available_models = {model_entry.model for model_entry in ollama.list().models}
    if model_id not in available_models:
        print(f"\nDownloading ollama model '{model_id}'..")
        pull_ollama_model(model_id)
        print(f"\nDownloaded ollama model '{model_id}'")


def pull_ollama_model(model_id: str):
    """Pull a ollama model. Show a progress bar.

    Based on
    https://github.com/ollama/ollama-python/blob/main/examples/pull.py

    """
    current_digest, bars = "", {}
    for progress in ollama.pull(model_id, stream=True):
        digest = progress.get("digest", "")
        if digest != current_digest and current_digest in bars:
            bars[current_digest].close()

        if digest not in bars and (total := progress.get("total")):
            bars[digest] = tqdm(
                total=total,
                desc=f"Pulling {digest[7:19]}",
                unit="B",
                unit_scale=True,
            )
            current_digest = digest

        if completed := progress.get("completed"):
            bars[digest].update(completed - bars[digest].n)

        current_digest = digest
    return
