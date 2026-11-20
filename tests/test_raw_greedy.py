import torch

from emotion_neurons.decomposition import greedy
from emotion_neurons.decomposition.raw_greedy import first_reaching, raw_greedy


def test_batched_raw_greedy_matches_the_gram_greedy_per_token():
    torch.manual_seed(0)
    W = torch.randn(16, 40, dtype=torch.float64)
    a = torch.randn(3, 40, dtype=torch.float64)
    d = torch.randn(3, 16, dtype=torch.float64)
    run = raw_greedy(W, W.T @ W, a, d, steps=12)
    for i in range(3):
        G = (W.T @ W) * a[i, :, None] * a[i, None, :]
        c = a[i] * (W.T @ (d[i] / d[i].norm()))
        order, cos = greedy(G, c, 12)
        assert run.order[i].tolist() == order
        torch.testing.assert_close(run.cos[i], torch.tensor(cos, dtype=torch.float64))


def test_cos_is_that_of_the_summed_raw_terms():
    torch.manual_seed(1)
    W = torch.randn(8, 30)
    a = torch.randn(2, 30)
    d = torch.randn(2, 8)
    run = raw_greedy(W, W.T @ W, a, d, steps=5)
    for i in range(2):
        s = W[:, run.order[i]] @ a[i, run.order[i]]
        expected = torch.nn.functional.cosine_similarity(s, d[i], dim=0)
        torch.testing.assert_close(run.cos[i, -1], expected, atol=1e-5, rtol=1e-4)


def test_first_reaching_gives_zero_when_never_reached():
    cos = torch.tensor([[0.1, 0.6, 0.8], [0.1, 0.2, 0.3]])
    assert first_reaching(cos, 0.5).tolist() == [2, 0]


def test_allowed_restricts_the_candidates_and_stops_when_none_are_left():
    torch.manual_seed(2)
    W = torch.randn(8, 20)
    a = torch.randn(2, 20)
    d = torch.randn(2, 8)
    allowed = torch.zeros(2, 20, dtype=torch.bool)
    allowed[:, :4] = True
    run = raw_greedy(W, W.T @ W, a, d, steps=6, allowed=allowed)
    assert set(run.order[:, :4].flatten().tolist()) <= set(range(4))
    assert (run.order[:, 4:] == -1).all()
    torch.testing.assert_close(run.cos[:, 4], run.cos[:, 3])
    s = W[:, :4] @ a[0, :4]
    torch.testing.assert_close(run.cos[0, 3], torch.nn.functional.cosine_similarity(s, d[0], dim=0), atol=1e-5, rtol=1e-4)
