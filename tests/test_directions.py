import pytest
import torch

from emotion_neurons.directions import ablation_hooks, addition_hooks, hooked
from emotion_neurons.models import decoder_layers

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"


@pytest.fixture(scope="module")
def model():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(TINY, local_files_only=True, dtype=torch.float32).eval()
    except OSError:
        pytest.skip("tiny Qwen3 test model is not in the local cache")


def layer_inputs(model, ids):
    seen = {}
    handles = [
        b.register_forward_pre_hook(
            lambda m, a, k, i=i: seen.__setitem__(i, (a[0] if a else k["hidden_states"]).clone()), with_kwargs=True
        )
        for i, b in enumerate(decoder_layers(model))
    ]
    with torch.no_grad(), hooked(handles):
        model(input_ids=ids)
    return seen


def test_ablation_removes_the_direction_from_every_layer_input(model):
    ids = torch.tensor([[1, 5, 9, 13, 17]])
    unit = torch.randn(model.config.hidden_size)
    unit = unit / unit.norm()
    with hooked(ablation_hooks(model, unit)):
        seen = layer_inputs(model, ids)
    for i in range(1, len(seen)):  # layer 0's input is the embedding, ablated by layer 0's own pre-hook (after capture)
        assert (seen[i] @ unit).abs().max() < 1e-4


def test_addition_shifts_its_layer_input_and_hooks_come_off(model):
    ids = torch.tensor([[2, 4, 6, 8]])
    vector = torch.randn(model.config.hidden_size)
    plain = layer_inputs(model, ids)
    with hooked(addition_hooks(model, 1, vector)):
        with torch.no_grad():
            seen = {}
            handle = decoder_layers(model)[1].register_forward_pre_hook(
                lambda m, a, k: seen.__setitem__("x", (a[0] if a else k["hidden_states"]).clone()), with_kwargs=True
            )
            model(input_ids=ids)
            handle.remove()
    torch.testing.assert_close(seen["x"], plain[1] + vector)
    torch.testing.assert_close(layer_inputs(model, ids)[1], plain[1])  # the addition hook is gone
