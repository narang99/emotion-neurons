"""One model per process, shared by the steps of a run."""

from functools import cache

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..config import ModelConfig
from ..models import load_model


@cache
def loaded_model(model: ModelConfig, device: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    return load_model(model, device)
