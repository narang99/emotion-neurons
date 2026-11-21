import torch

from emotion_neurons.decomposition.residual import first_below, residual_greedy


def test_leftover_is_the_norm_of_the_neurons_outside_the_set():
    torch.manual_seed(0)
    W = torch.randn(12, 40, dtype=torch.float64)
    shift = torch.randn(40, dtype=torch.float64)
    run = residual_greedy(W, W.T @ W, shift, steps=15)
    d = W @ shift
    for k in (1, 5, len(run.order)):
        S = run.order[:k]
        rest = torch.ones(40, dtype=torch.bool)
        rest[S] = False
        expected = (W[:, rest] @ shift[rest]).norm() / d.norm()
        torch.testing.assert_close(run.leftover[k - 1], expected.float())


def test_each_step_takes_the_best_neuron():
    torch.manual_seed(1)
    W = torch.randn(6, 10, dtype=torch.float64)
    shift = torch.randn(10, dtype=torch.float64)
    run = residual_greedy(W, W.T @ W, shift, steps=3)
    d = W @ shift
    S: list[int] = []
    for j in run.order.tolist():
        r = d - W[:, S] @ shift[S]
        best = min((n for n in range(10) if n not in S), key=lambda n: float((r - W[:, n] * shift[n]).norm()))
        assert j == best
        S.append(j)


def test_first_below():
    assert first_below(torch.tensor([0.9, 0.5, 0.2]), 0.5) == 2
    assert first_below(torch.tensor([0.9, 0.8]), 0.5) is None
