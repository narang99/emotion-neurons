import torch

from emotion_neurons.decomposition.signal_noise import classify, neurons_for_mass, noise_spread


def test_classify_separates_signal_constant_and_noise():
    torch.manual_seed(0)
    own = torch.randn(200, 6) * 0.1
    other = torch.randn(300, 6) * 0.1
    own[:, 0] += 1.0  # specific: pushes on own stories only
    own[:, 1] += 1.0  # constant: pushes everywhere
    other[:, 1] += 1.0
    g = classify(own, other, t_cut=4.6)
    assert g.specific.tolist() == [True, False, False, False, False, False]
    assert g.constant.tolist() == [False, True, False, False, False, False]
    assert g.noise[2:].all()
    assert (g.signal == (g.specific | g.constant)).all()


def test_neurons_for_mass_counts_largest_first():
    values = torch.tensor([5.0, -1.0, 3.0, 0.5, 2.5])
    members = torch.tensor([True, True, True, False, True])
    assert neurons_for_mass(values, members, 0.8) == 2  # 5 + 3 = 8 >= 0.8 * 10


def test_noise_spread_of_independent_noise_matches_prediction():
    torch.manual_seed(1)
    x = torch.randn(4000, 50)
    s = noise_spread(x, x, torch.ones(50, dtype=torch.bool))
    assert abs(s["sd_train"] / s["sd_if_independent"] - 1) < 0.05
