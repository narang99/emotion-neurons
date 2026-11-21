import torch

from emotion_neurons.sparse import lipschitz, residual_greedy, scaled_columns, set_leftover, solve_mask, support
from emotion_neurons.sparse.relaxed import leftover, prox_step

torch.manual_seed(0)
W = torch.randn(16, 40, dtype=torch.float64)
SHIFT = torch.randn(40, dtype=torch.float64)


def test_greedy_leftover_matches_the_set_it_picked():
    run = residual_greedy(W, W.T @ W, SHIFT, steps=10)
    for k in (1, 5, len(run.order)):
        members = torch.zeros(40, dtype=torch.bool)
        members[run.order[:k]] = True
        assert abs(run.leftover[k - 1].item() - set_leftover(W, SHIFT, members)) < 1e-10


def test_greedy_first_step_is_the_best_single_neuron():
    run = residual_greedy(W, W.T @ W, SHIFT, steps=1)
    singles = [set_leftover(W, SHIFT, torch.arange(40) == j) for j in range(40)]
    assert run.order[0].item() == min(range(40), key=singles.__getitem__)


def test_full_mask_rebuilds_the_write_exactly():
    B, d = scaled_columns(W, SHIFT)
    assert leftover(B, d, torch.ones(40, dtype=torch.float64)) < 1e-12


def test_mask_without_penalty_reaches_zero_leftover():
    B, d = scaled_columns(W, SHIFT)
    solution = solve_mask(B, d, 0.0, lipschitz(B), torch.zeros(40, dtype=torch.float64), 20000, 1e-9)
    assert leftover(B, d, solution.mask) < 1e-4


def test_mask_is_a_fixed_point_of_the_prox_step():
    B, d = scaled_columns(W, SHIFT)
    L = lipschitz(B)
    solution = solve_mask(B, d, 0.05, L, torch.zeros(40, dtype=torch.float64), 20000, 1e-10)
    m = solution.mask
    assert ((m >= 0) & (m <= 1)).all()
    torch.testing.assert_close(prox_step(B, d, m, 0.05, L), m, atol=1e-8, rtol=0)
    assert 0 < support(m).sum() < 40


def test_prefix_errors_match_set_error_of_each_prefix():
    from emotion_neurons.sparse import prefix_errors, set_error

    order = torch.randperm(40)[:12]
    target = torch.randn(16, dtype=torch.float64)
    curve = prefix_errors(W, SHIFT, order, target)
    for k in (1, 6, 12):
        members = torch.zeros(40, dtype=torch.bool)
        members[order[:k]] = True
        assert abs(curve[k - 1].item() - set_error(W, SHIFT, members, target)) < 1e-12


def test_folds_hold_out_each_story_once():
    from emotion_neurons.sparse import fold_split
    from emotion_neurons.texts import StorySplit

    label = torch.tensor([0] * 6 + [1] * 6 + [-1] * 12)
    story_idx = torch.tensor(list(range(6)) * 2 + list(range(12)))
    base = StorySplit(label=label, heldout=torch.zeros(24, dtype=torch.bool), topic=torch.zeros(24, dtype=torch.long))
    held = torch.stack([fold_split(base, story_idx, k).heldout for k in range(6)])
    assert (held.sum(0) == 1).all()
    assert held.sum(1).tolist() == [4] * 6
