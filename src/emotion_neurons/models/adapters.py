"""Model families: how to load a model and its tokenizer, how to wrap an instruction in its chat format, and the
checkpoint names of the tensors read one at a time (`models.weights`).

- The family comes from the config (`config.ModelConfig`), and so does the revision.
- `hf` (Qwen3 and other native transformers models): AutoModelForCausalLM, the tokenizer's chat template.
- `qwen1` (Qwen/Qwen-7B-Chat): converted to a native Qwen2 model (`models.qwen1`), ChatML by hand.
- Both end up with Qwen3-style layer names (model.layers, mlp.down_proj, post_attention_layernorm), so the hooks
  in `models.qwen3` read every family unchanged.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from ..config import ModelConfig
from . import qwen1

TensorKind = str
"""One of "down_proj" (needs a layer), "lm_head", "final_norm"."""


@dataclass(frozen=True)
class ModelAdapter:
    load: Callable[[str, str | None, str], tuple[PreTrainedModel, PreTrainedTokenizerBase]]
    load_tokenizer: Callable[[str, str | None], PreTrainedTokenizerBase]
    chat_prompt: Callable[[PreTrainedTokenizerBase, str], str]
    tensor_name: Callable[[TensorKind, int | None], str]


def hf_tokenizer(name: str, revision: str | None) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, token=os.environ.get("HF_TOKEN") or None)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def hf_load(name: str, revision: str | None, device: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """bf16 and right padding, as every run here assumes."""
    token = os.environ.get("HF_TOKEN") or None
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch.bfloat16, device_map=device, token=token)
    return model.eval(), hf_tokenizer(name, revision)


def hf_chat_prompt(tokenizer: PreTrainedTokenizerBase, instruction: str) -> str:
    """The model's own chat template, with the assistant turn opened; Qwen3's thinking turned off."""
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def hf_tensor_name(kind: TensorKind, layer: int | None = None) -> str:
    return {
        "down_proj": f"model.layers.{layer}.mlp.down_proj.weight",
        "lm_head": "lm_head.weight",
        "final_norm": "model.norm.weight",
    }[kind]


ADAPTERS: dict[str, ModelAdapter] = {
    "hf": ModelAdapter(hf_load, hf_tokenizer, hf_chat_prompt, hf_tensor_name),
    "qwen1": ModelAdapter(
        qwen1.load_qwen1, qwen1.load_qwen1_tokenizer, lambda _tok, text: qwen1.chat_prompt(text), qwen1.checkpoint_name
    ),
}


def adapter_for(model: ModelConfig) -> ModelAdapter:
    return ADAPTERS[model.family]


def load_model(model: ModelConfig, device: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    return adapter_for(model).load(model.name, model.revision, device)


def load_tokenizer(model: ModelConfig) -> PreTrainedTokenizerBase:
    return adapter_for(model).load_tokenizer(model.name, model.revision)
