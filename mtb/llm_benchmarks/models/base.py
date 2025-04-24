from dataclasses import dataclass
from typing import Callable


@dataclass
class ModelSpec:
    # identifier for the model
    name: str = None
    # number of parameters in billions
    num_params: float = None
    # Function that formats the prompt
    prompt_formatter: Callable = None
    # model_id for each framework and dtype
    model_ids = dict()
