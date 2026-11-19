import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from emotion_neurons.interventions import NeuronEdit, run_with_edits

LAYER = 1


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
    )  # fmt: skip
    return Qwen3ForCausalLM(config).eval(), torch.randint(0, 64, (1, 10))


def activations(model, ids):
    store = {}
    handle = model.model.layers[LAYER].mlp.down_proj.register_forward_pre_hook(lambda _m, a: store.__setitem__("a", a[0]))
    with torch.no_grad():
        model(input_ids=ids)
    handle.remove()
    return store["a"][0]


def test_adding_a_shift_moves_the_layer_output_by_its_down_projection(tiny):
    model, ids = tiny
    neurons = torch.tensor([3, 7, 20])
    shift = torch.tensor([0.5, -1.0, 2.0])
    clean = run_with_edits(model, ids, [], [LAYER], (0, 10)).stream_means[0]
    edited = run_with_edits(model, ids, [NeuronEdit(layer=LAYER, neurons=neurons, values=shift, kind="add")], [LAYER], (0, 10))
    expected = model.model.layers[LAYER].mlp.down_proj.weight[:, neurons] @ shift
    torch.testing.assert_close(edited.stream_means[0] - clean, expected, atol=1e-5, rtol=1e-4)


def test_holding_neurons_at_their_own_values_changes_nothing(tiny):
    model, ids = tiny
    neurons = torch.tensor([1, 2])
    clean = run_with_edits(model, ids, [], [2], (0, 10))
    a = activations(model, ids)[:, neurons]
    held_first = NeuronEdit(layer=LAYER, neurons=neurons, values=a[0], kind="hold")
    edited = run_with_edits(model, ids[:, :1], [held_first], [2], (0, 1))
    reference = run_with_edits(model, ids[:, :1], [], [2], (0, 1))
    torch.testing.assert_close(edited.last_logits, reference.last_logits)
    assert clean.last_logits.shape == (64,)


def test_clip_keeps_activations_inside_the_bounds(tiny):
    model, ids = tiny
    neurons = torch.arange(48)
    lo, hi = torch.full((48,), -0.01), torch.full((48,), 0.01)
    store = {}
    edit = NeuronEdit(layer=LAYER, neurons=neurons, values=lo, kind="clip", upper=hi)
    handle = model.model.layers[LAYER].mlp.down_proj.register_forward_pre_hook(edit.hook())
    seen = model.model.layers[LAYER].mlp.down_proj.register_forward_pre_hook(lambda _m, a: store.__setitem__("a", a[0]))
    with torch.no_grad():
        model(input_ids=ids)
    handle.remove()
    seen.remove()
    assert store["a"].abs().max() <= 0.01 + 1e-6
