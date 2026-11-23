"""Model families: how to load a model and its tokenizer, how to wrap an instruction in its chat format, and the
checkpoint names of the tensors read one at a time (`models.weights`, `lens`).

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

from . import qwen1

TensorKind = str
"""One of "down_proj" (needs a layer), "lm_head", "final_norm"."""


@dataclass(frozen=True)
class ModelAdapter:
    family: str
    load: Callable[[str, str], tuple[PreTrainedModel, PreTrainedTokenizerBase]]
    load_tokenizer: Callable[[str], PreTrainedTokenizerBase]
    chat_prompt: Callable[[PreTrainedTokenizerBase, str], str]
    tensor_name: Callable[[TensorKind, int | None], str]
    revision: Callable[[str], str | None]


def hf_tokenizer(name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(name, token=os.environ.get("HF_TOKEN") or None)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def hf_load(name: str, device: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """bf16 and right padding, as every run here assumes."""
    token = os.environ.get("HF_TOKEN") or None
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device, token=token)
    return model.eval(), hf_tokenizer(name)


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


HF = ModelAdapter("hf", hf_load, hf_tokenizer, hf_chat_prompt, hf_tensor_name, revision=lambda _name: None)
QWEN1 = ModelAdapter(
    "qwen1", qwen1.load_qwen1, qwen1.load_qwen1_tokenizer, lambda _tok, text: qwen1.chat_prompt(text),
    qwen1.checkpoint_name, revision=lambda name: qwen1.REVISIONS[name],
)  # fmt: skip


def adapter_for(name: str) -> ModelAdapter:
    return QWEN1 if qwen1.is_qwen1(name) else HF


def load_model(name: str, device: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    return adapter_for(name).load(name, device)


def load_tokenizer(name: str) -> PreTrainedTokenizerBase:
    return adapter_for(name).load_tokenizer(name)
