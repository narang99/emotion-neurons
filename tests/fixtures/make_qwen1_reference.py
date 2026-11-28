"""Builds the tiny Qwen-1 reference for tests/test_qwen1.py, with Qwen-1's own 2023 model code.

It needs an old transformers, so it runs in its own environment:

    uv run --isolated --python 3.11 --with "transformers==4.40.2" --with "torch==2.3.1" --with "numpy<2" \
        --with tiktoken --with einops --with transformers_stream_generator --with safetensors \
        python tests/fixtures/make_qwen1_reference.py

- A random Qwen-1 with tiny sizes (2 layers, hidden 64, vocab 1000), fixed seed.
- Saves its weights (the original Qwen-1 names), its config, a fixed input and the logits it gives:
  tests/fixtures/qwen1_tiny/{model.safetensors, config.json, reference.safetensors}.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

REPO, REVISION = "Qwen/Qwen-7B-Chat", "93a65d34827a3cc269b727e67004743b723e2f83"
OUT = Path(__file__).parent / "qwen1_tiny"
SIZES = dict(
    num_hidden_layers=2,
    hidden_size=64,
    intermediate_size=256,
    num_attention_heads=4,
    kv_channels=16,
    vocab_size=1000,
    seq_length=2048,
    use_flash_attn=False,
    bf16=False,
    fp16=False,
    fp32=True,
)


def main():
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(REPO, revision=REVISION, trust_remote_code=True, **SIZES)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.float32).eval()
    with torch.no_grad():  # random biases too, so the conversion of every bias is checked
        for name, p in model.named_parameters():
            p.normal_(0, 0.05) if not name.endswith(("ln_1.weight", "ln_2.weight", "ln_f.weight")) else p.uniform_(0.5, 1.5)
    input_ids = torch.randint(0, SIZES["vocab_size"], (2, 12), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    OUT.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(OUT / "model.safetensors"))
    (OUT / "config.json").write_text(json.dumps(config.to_dict(), indent=1, default=str))
    save_file({"input_ids": input_ids, "logits": logits.contiguous()}, str(OUT / "reference.safetensors"))
    print(f"saved {OUT}: logits {tuple(logits.shape)}, {sum(p.numel() for p in model.parameters())} parameters")


if __name__ == "__main__":
    main()
