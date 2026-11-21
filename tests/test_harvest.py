from pathlib import Path

import torch

from emotion_neurons.harvest import CorpusActs, band_masks, band_names, neurons_until, window_text


def test_band_names_cover_both_poles():
    assert band_names([0.97, 0.99, 0.999, 0.9999]) == ["p97-p99", "p99-p99.9", "p99.9-p99.99", "p99.99+"]
    assert band_names([0.03, 0.01, 0.001, 0.0001]) == ["p3-p1", "p1-p0.1", "p0.1-p0.01", "p0.01-"]


def test_bands_split_each_pole_without_overlap():
    a = torch.tensor([-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])
    high = band_masks(a, torch.tensor([1.0, 2.0]), "high")
    low = band_masks(a, torch.tensor([-1.0, -2.0]), "low")
    assert [m.nonzero().squeeze(1).tolist() for m in high] == [[4], [5, 6]]
    assert [m.nonzero().squeeze(1).tolist() for m in low] == [[2], [0, 1]]


def test_neurons_until_stops_at_the_share(tmp_path: Path):
    csv = tmp_path / "set.csv"
    csv.write_text("rank,neuron,rebuilt_after_pct\n1,7,40\n2,3,74.9\n3,9,75.0\n4,1,80\n")
    assert neurons_until(csv, 75) == [7, 3, 9]


def test_window_text_stays_inside_its_window():
    class Tok:
        def decode(self, ids):
            return "".join(chr(97 + int(i)) for i in ids)

    corpus = CorpusActs(
        acts=torch.zeros(6, 1), tokens=torch.arange(6), window=torch.tensor([0, 0, 0, 1, 1, 1]),
        position=torch.tensor([0, 1, 2, 0, 1, 2]), sources=[("x", 0), ("x", 1)],
    )  # fmt: skip
    bounds = corpus.window_bounds()
    assert bounds.tolist() == [[0, 3], [3, 6]]
    assert window_text(Tok(), corpus, bounds, 4, before=5, after=5) == "d[[e]]f"


def test_batch_groups_respect_neuron_and_cluster_budgets():
    from emotion_neurons.config import ModesConfig
    from emotion_neurons.modes import batch_groups

    sizes = {1: 30, 2: 30, 3: 30, 4: 200, 5: 5, 6: 5}
    result = {n: {"high": {"clusters": [0] * (s // 2)}, "low": {"clusters": [0] * (s - s // 2)}} for n, s in sizes.items()}
    groups = batch_groups(result, list(sizes), ModesConfig(neurons_per_batch=8, clusters_per_batch=80))
    assert groups == [[1, 2], [3], [4], [5, 6]]
    assert batch_groups(result, [5, 6], ModesConfig(neurons_per_batch=1)) == [[5], [6]]
