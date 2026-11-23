"""Qwen-1 chat models (e.g. Qwen/Qwen-7B-Chat, as in the refusal-direction paper) as native transformers Qwen2 models.

- Qwen-1 ships 2023 model code (trust_remote_code) that no longer runs on transformers 5 (removed APIs).
- Its architecture is Qwen2's: RMSNorm, RoPE, attention with q/k/v bias and no output bias, a SwiGLU MLP.
  So its weights map one to one onto a Qwen2 model:
  - attn.c_attn (q, k and v stacked) -> self_attn.q_proj, k_proj, v_proj; attn.c_proj -> self_attn.o_proj;
  - mlp: Qwen-1 computes w1(x) * silu(w2(x)), so w2 -> gate_proj, w1 -> up_proj; mlp.c_proj -> down_proj;
  - ln_1 -> input_layernorm, ln_2 -> post_attention_layernorm, wte -> embed_tokens, ln_f -> norm.
- Exact for sequences up to Qwen-1's `seq_length` (2048): its dynamic NTK and log-n attention act only beyond it.
  tests/test_qwen1.py checks the logits against Qwen-1's own code on a tiny random model.
- The converted model has Qwen3's layer names, so the hooks in `models.qwen3` read it unchanged.
- The tokenizer is Qwen-1's own (tiktoken based), loaded with trust_remote_code at a pinned revision.
"""

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Qwen2Config, Qwen2ForCausalLM

REVISIONS = {"Qwen/Qwen-7B-Chat": "93a65d34827a3cc269b727e67004743b723e2f83"}
"""Pinned revisions: the repo's remote tokenizer code is downloaded and run, so it must not change under us."""


def is_qwen1(name: str) -> bool:
    return name in REVISIONS


def qwen2_config(q1: dict) -> Qwen2Config:
    heads, hidden = q1["num_attention_heads"], q1["hidden_size"]
    assert heads * q1["kv_channels"] == hidden, "Qwen-1 head size must equal hidden / heads"
    assert q1.get("rotary_pct", 1.0) == 1.0, "partial rotary embeddings are not mapped"
    return Qwen2Config(
        vocab_size=q1["vocab_size"], hidden_size=hidden, intermediate_size=q1["intermediate_size"] // 2,
        num_hidden_layers=q1["num_hidden_layers"], num_attention_heads=heads, num_key_value_heads=heads,
        hidden_act="silu", rms_norm_eps=q1["layer_norm_epsilon"], max_position_embeddings=q1["seq_length"],
        rope_parameters={"rope_type": "default", "rope_theta": float(q1["rotary_emb_base"])},
        tie_word_embeddings=False, use_sliding_window=False, attention_dropout=0.0,
    )  # fmt: skip


def qwen2_tensors(name: str, tensor: Tensor) -> dict[str, Tensor]:
    """One Qwen-1 checkpoint tensor as the Qwen2 tensor(s) it becomes."""
    if name == "transformer.wte.weight":
        return {"model.embed_tokens.weight": tensor}
    if name == "transformer.ln_f.weight":
        return {"model.norm.weight": tensor}
    if name == "lm_head.weight":
        return {"lm_head.weight": tensor}
    prefix, layer, rest = name.split(".", 3)[0:1], name.split(".")[2], name.split(".", 3)[3]
    assert prefix == ["transformer"], name
    out = f"model.layers.{layer}."
    if rest.startswith("attn.c_attn."):
        kind = rest.rsplit(".", 1)[1]  # weight or bias
        q, k, v = tensor.chunk(3, dim=0)
        return {f"{out}self_attn.q_proj.{kind}": q, f"{out}self_attn.k_proj.{kind}": k, f"{out}self_attn.v_proj.{kind}": v}
    simple = {
        "attn.c_proj.weight": "self_attn.o_proj.weight", "ln_1.weight": "input_layernorm.weight",
        "ln_2.weight": "post_attention_layernorm.weight", "mlp.w2.weight": "mlp.gate_proj.weight",
        "mlp.w1.weight": "mlp.up_proj.weight", "mlp.c_proj.weight": "mlp.down_proj.weight",
    }  # fmt: skip
    assert rest in simple, f"unmapped Qwen-1 tensor {name}"
    return {out + simple[rest]: tensor}


def checkpoint_name(kind: str, layer: int | None = None) -> str:
    """Qwen-1 checkpoint names, for single-tensor reads (`models.weights`)."""
    return {
        "down_proj": f"transformer.h.{layer}.mlp.c_proj.weight",
        "lm_head": "lm_head.weight",
        "final_norm": "transformer.ln_f.weight",
    }[kind]


def convert(config: dict, shards: list[Path], dtype: torch.dtype) -> Qwen2ForCausalLM:
    with torch.device("meta"):
        model = Qwen2ForCausalLM(qwen2_config(config))
    state = {}
    for shard in shards:
        with safe_open(str(shard), "pt") as f:
            for name in f.keys():
                state.update({k: v.to(dtype) for k, v in qwen2_tensors(name, f.get_tensor(name)).items()})
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    assert not unexpected, f"unexpected tensors: {unexpected[:5]}"
    assert not [m for m in missing if not m.endswith("rotary_emb.inv_freq")], f"missing tensors: {missing[:5]}"
    model.model.rotary_emb.__init__(model.config)  # inv_freq is a buffer built at init, not a checkpoint tensor
    return model.eval()


def load_qwen1(name: str, device: str) -> tuple[Qwen2ForCausalLM, PreTrainedTokenizerBase]:
    revision = REVISIONS[name]
    config = json.loads(Path(hf_hub_download(name, "config.json", revision=revision)).read_text())
    index = json.loads(Path(hf_hub_download(name, "model.safetensors.index.json", revision=revision)).read_text())
    shards = [Path(hf_hub_download(name, f, revision=revision)) for f in sorted(set(index["weight_map"].values()))]
    model = convert(config, shards, torch.bfloat16).to(device)
    return model, load_qwen1_tokenizer(name)


def load_qwen1_tokenizer(name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(name, revision=REVISIONS[name], trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eod_id  # Qwen-1 has no pad token; <|endoftext|> is never inside a prompt
    return tokenizer


CHAT_SYSTEM = "You are a helpful assistant."


def chat_prompt(instruction: str) -> str:
    """Qwen-1's ChatML format (its generation utilities' default system prompt). The last five tokens,
    <|im_end|> \\n <|im_start|> assistant \\n, are the post-instruction positions."""
    return f"<|im_start|>system\n{CHAT_SYSTEM}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
