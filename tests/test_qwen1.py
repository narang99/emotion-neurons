import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from emotion_neurons.models.qwen1 import convert

FIXTURE = Path(__file__).parent / "fixtures" / "qwen1_tiny"


def test_converted_qwen1_gives_the_original_logits():
    """Logits of a tiny random Qwen-1, from its own 2023 code (fixtures/make_qwen1_reference.py), against the
    same weights converted to a native Qwen2 model."""
    config = json.loads((FIXTURE / "config.json").read_text())
    model = convert(config, [FIXTURE / "model.safetensors"], torch.float32)
    reference = load_file(str(FIXTURE / "reference.safetensors"))
    with torch.no_grad():
        logits = model(input_ids=reference["input_ids"]).logits
    torch.testing.assert_close(logits, reference["logits"], atol=1e-4, rtol=1e-4)


def test_the_mlp_is_swiglu_with_gate_from_w2():
    """Qwen-1 computes w1(x) * silu(w2(x)): w2 must land in gate_proj, w1 in up_proj."""
    config = json.loads((FIXTURE / "config.json").read_text())
    model = convert(config, [FIXTURE / "model.safetensors"], torch.float32)
    original = load_file(str(FIXTURE / "model.safetensors"))
    mlp = model.model.layers[0].mlp
    torch.testing.assert_close(mlp.gate_proj.weight, original["transformer.h.0.mlp.w2.weight"])
    torch.testing.assert_close(mlp.up_proj.weight, original["transformer.h.0.mlp.w1.weight"])
    assert mlp.down_proj.weight.shape[1] == config["intermediate_size"] // 2
