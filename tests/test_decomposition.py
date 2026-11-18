import itertools

import pytest
import torch

from emotion_neurons.decomposition import SetState, greedy, neuron_terms
from emotion_neurons.decomposition.dims_then_neurons import cover_count
from emotion_neurons.decomposition.selection import gram, plain

torch.manual_seed(0)
D_MODEL, D_MLP = 16, 24


@pytest.fixture
def problem():
    W = torch.randn(D_MODEL, D_MLP, dtype=torch.float64)
    a = torch.randn(D_MLP, dtype=torch.float64)
    G = gram(W.T @ W, a)
    c = plain(W, a)
    return W, a, G, c


def direct_cos(W, a, members: list[int]) -> float:
    d = W @ a
    s = W[:, members] @ a[members]
    return float(torch.nn.functional.cosine_similarity(s, d, dim=0))


def test_neuron_terms_sum_to_the_write(problem):
    W, a, _, _ = problem
    torch.testing.assert_close(neuron_terms(W, a).sum(1), W @ a)


def test_set_state_matches_direct_cosine_under_add_and_remove(problem):
    W, a, G, c = problem
    state = SetState(G, c, [3, 7])
    state.add(11)
    state.remove(3)
    state.add(0)
    assert state.cos == pytest.approx(direct_cos(W, a, [7, 11, 0]), abs=1e-10)


def test_greedy_picks_the_best_addition_each_step(problem):
    W, a, G, c = problem
    order, cos = greedy(G, c, steps=5)
    for k in range(5):
        best = max((n for n in range(D_MLP) if n not in order[:k]), key=lambda n: direct_cos(W, a, order[:k] + [n]))
        assert order[k] == best
        assert cos[k] == pytest.approx(direct_cos(W, a, order[: k + 1]), abs=1e-10)


def test_best_swap_is_the_best_one_swap(problem):
    W, a, G, c = problem
    members = [1, 4, 9]
    out, into, cos = SetState(G, c, members).best_swap()
    swaps = [(i, j) for i, j in itertools.product(members, range(D_MLP)) if j not in members]
    best = max(swaps, key=lambda ij: direct_cos(W, a, [m for m in members if m != ij[0]] + [ij[1]]))
    assert (out, into) == best
    assert cos == pytest.approx(direct_cos(W, a, [m for m in members if m != out] + [into]), abs=1e-10)


def test_cover_count_reaches_the_share():
    values = torch.tensor([[5.0, 3.0, 1.0, 1.0]])
    assert int(cover_count(values, 0.5)) == 1
    assert int(cover_count(values, 0.8)) == 2
    assert int(cover_count(values, 1.0)) == 4
