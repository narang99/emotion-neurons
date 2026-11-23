import torch

from emotion_neurons.sparse import first_reaching, greedy_set

torch.manual_seed(0)
W = torch.randn(16, 40, dtype=torch.float64)
SHIFT = torch.randn(40, dtype=torch.float64)


def rebuilt_by(members: torch.Tensor) -> float:
    d = W @ SHIFT
    missing = W[:, ~members] @ SHIFT[~members]
    return float(1 - (missing @ missing) / (d @ d))


def members_of(order: torch.Tensor) -> torch.Tensor:
    members = torch.zeros(40, dtype=torch.bool)
    members[order] = True
    return members


def test_rebuilt_share_matches_the_set_greedy_picked():
    run = greedy_set(W, W.T @ W, SHIFT, steps=40)
    for k in (1, 5, len(run.order)):
        assert abs(run.rebuilt[k - 1].item() - rebuilt_by(members_of(run.order[:k]))) < 1e-10


def test_first_step_is_the_best_single_neuron():
    run = greedy_set(W, W.T @ W, SHIFT, steps=1)
    singles = [rebuilt_by(torch.arange(40) == j) for j in range(40)]
    assert run.order[0].item() == max(range(40), key=singles.__getitem__)


def test_all_neurons_rebuild_the_write_exactly():
    assert abs(rebuilt_by(torch.ones(40, dtype=torch.bool)) - 1) < 1e-12


def test_first_reaching_counts_from_one():
    assert first_reaching(torch.tensor([0.2, 0.5, 0.9, 0.95]), 0.9) == 3
    assert first_reaching(torch.tensor([0.2, 0.5]), 0.9) is None
