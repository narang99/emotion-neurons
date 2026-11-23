import torch

from emotion_neurons.poles import NEURON_CHUNK, padded_batch, quantiles_per_neuron

torch.manual_seed(0)


def test_quantiles_are_the_kth_smallest_value_per_neuron():
    acts = torch.randn(1000, NEURON_CHUNK + 7).half()
    q = quantiles_per_neuron(acts, [0.03, 0.97])
    for row, quantile in enumerate([0.03, 0.97]):
        k = int(quantile * len(acts))
        torch.testing.assert_close(q[row], acts.float().kthvalue(k, dim=0).values)


def test_padded_batch_pads_right_and_masks_the_padding():
    ids, mask = padded_batch([[5, 6, 7], [8]], pad_id=0)
    assert ids.tolist() == [[5, 6, 7], [8, 0, 0]]
    assert mask.tolist() == [[1, 1, 1], [1, 0, 0]]
