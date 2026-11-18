import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from emotion_neurons.circuits import attribute, direct_sum, input_edges, keep_mask
from emotion_neurons.circuits.graph import ablated_score

LAYER = 2
MEANS = torch.full((LAYER + 1, 48), 0.05)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
    )  # fmt: skip
    model = Qwen3ForCausalLM(config).eval()
    ids = torch.randint(0, 64, (1, 10))
    unit = torch.nn.functional.normalize(torch.randn(32), dim=0)
    return model, ids, unit


def test_direct_shares_sum_to_the_score(tiny):
    model, ids, unit = tiny
    attribution = attribute(model, ids, LAYER, unit, MEANS)
    torch.testing.assert_close(direct_sum(attribution), attribution.score, atol=1e-4, rtol=1e-4)


def test_total_effect_equals_direct_effect_at_the_target_layer(tiny):
    model, ids, unit = tiny
    attribution = attribute(model, ids, LAYER, unit, MEANS)
    torch.testing.assert_close(attribution.total[LAYER], attribution.direct[LAYER], atol=1e-5, rtol=1e-4)


def test_edges_sum_to_the_mlp_part_of_the_reader_input(tiny):
    model, ids, unit = tiny
    attribution = attribute(model, ids, LAYER, unit, MEANS)
    tokens = torch.ones(ids.shape[1], dtype=torch.bool)
    edges = input_edges(model, attribution, LAYER, 5, tokens)
    mlp_writes = sum(attribution.activation[k] @ model.model.layers[k].mlp.down_proj.weight.T for k in range(LAYER))
    norm = model.model.layers[LAYER].post_attention_layernorm.weight
    row = model.model.layers[LAYER].mlp.gate_proj.weight[5]
    expected = ((mlp_writes * attribution.inverse_rms[LAYER][:, None] * norm) @ row).sum()
    total = sum(v.sum() for v in edges.shares["gate"].values())
    assert float(total) == pytest.approx(float(expected), rel=1e-4, abs=1e-5)


def test_keeping_every_neuron_leaves_the_score_unchanged(tiny):
    model, ids, unit = tiny
    tokens = torch.ones(ids.shape[1], dtype=torch.bool)
    means = torch.zeros(LAYER + 1, 48)
    everything = {layer: torch.ones(48, dtype=torch.bool) for layer in range(LAYER + 1)}
    full = ablated_score(model, ids, LAYER, unit, means, None, tokens)
    assert ablated_score(model, ids, LAYER, unit, means, everything, tokens) == pytest.approx(full, rel=1e-5)
    assert ablated_score(model, ids, LAYER, unit, means, keep_mask([], LAYER + 1, 48), tokens) != pytest.approx(full)


def test_effect_from_mean_at_the_target_layer_is_the_shift_times_the_direct_coefficient(tiny):
    model, ids, unit = tiny
    attribution = attribute(model, ids, LAYER, unit, MEANS)
    coefficient = model.model.layers[LAYER].mlp.down_proj.weight.float().T @ unit
    expected = (attribution.activation[LAYER] - MEANS[LAYER]) * coefficient.detach()
    torch.testing.assert_close(attribution.total_from_mean[LAYER], expected, atol=1e-5, rtol=1e-4)


def test_a_neurons_input_splits_exactly_over_the_stream(tiny):
    from emotion_neurons.circuits.neuron import read_stream, split_input

    model, ids, _ = tiny
    stream = read_stream(model, ids, LAYER)
    tokens = torch.ones(ids.shape[1], dtype=torch.bool)
    for lane in ("gate", "up"):
        split = split_input(model, stream, LAYER, 7, lane, tokens)
        assert split.rebuilt() == pytest.approx(split.actual, rel=1e-4, abs=1e-5)


def test_reader_shares_match_the_split_of_the_reader(tiny):
    from emotion_neurons.circuits.neuron import read_stream, reader_shares, split_input

    model, ids, _ = tiny
    stream = read_stream(model, ids, LAYER)
    tokens = torch.ones(ids.shape[1], dtype=torch.bool)
    shares = reader_shares(model, stream, 0, 11, LAYER, tokens)
    split = split_input(model, stream, LAYER, 3, "gate", tokens)
    assert float(shares["gate"][3]) == pytest.approx(float(split.neurons[0][11]), rel=1e-4, abs=1e-6)
