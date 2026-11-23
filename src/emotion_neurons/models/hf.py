"""Native transformers models (Qwen3 and others): AutoModelForCausalLM and the tokenizer's own chat template."""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from ..config import HFModel


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


class HFAdapter:
    def __init__(self, cfg: HFModel):
        self.cfg = cfg

    def load(self, device: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """bf16 and right padding, as every run here assumes."""
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.name, revision=self.cfg.revision, dtype=torch.bfloat16, device_map=device, token=hf_token()
        )
        return model.eval(), self.load_tokenizer()

    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.name, revision=self.cfg.revision, token=hf_token())
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def chat_prompt(self, tokenizer: PreTrainedTokenizerBase, instruction: str) -> str:
        """The model's own chat template, with the assistant turn opened; Qwen3's thinking turned off."""
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def tensor_name(self, kind: str, layer: int | None = None) -> str:
        return {
            "down_proj": f"model.layers.{layer}.mlp.down_proj.weight",
            "lm_head": "lm_head.weight",
            "final_norm": "model.norm.weight",
        }[kind]
