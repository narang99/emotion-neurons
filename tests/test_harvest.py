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


def banded_corpus() -> tuple[CorpusActs, dict]:
    """One neuron; high-pole bands of 40, 20 and 5 tokens past edges 1, 2, 3; low-pole band of 30 below -1."""
    a = torch.cat(
        [torch.zeros(100), torch.full((40,), 1.5), torch.full((20,), 2.5), torch.full((5,), 3.5), -torch.full((30,), 2.0)]
    )
    n = len(a)
    corpus = CorpusActs(
        acts=a[:, None].half(), tokens=torch.arange(n), window=torch.zeros(n, dtype=torch.long),
        position=torch.arange(n), sources=[("x", 0)],
    )  # fmt: skip
    edges = {"high": torch.tensor([[1.0], [2.0], [3.0]]), "low": torch.tensor([[-1.0], [-3.0], [-4.0]])}
    return corpus, edges


def test_per_band_hits_take_up_to_k_from_each_band():
    from emotion_neurons.config import HarvestConfig
    from emotion_neurons.harvest import band_index, pole_hits

    corpus, edges = banded_corpus()
    hits = pole_hits(corpus, 0, edges, HarvestConfig(hits_per_band=10), torch.Generator().manual_seed(0))
    a = corpus.acts[:, 0].float()
    high = band_index(a, edges["high"][:, 0], "high")[hits["high"]]
    assert torch.bincount(high, minlength=3).tolist() == [10, 10, 5]
    assert torch.bincount(band_index(a, edges["low"][:, 0], "low")[hits["low"]], minlength=3).tolist() == [10, 0, 0]


def test_hit_weights_give_back_each_band_size_and_shares_sum_to_100():
    import numpy as np

    from emotion_neurons.config import HarvestConfig, ModesConfig
    from emotion_neurons.harvest import pole_hits
    from emotion_neurons.modes import hit_bands, pole_modes

    corpus, edges = banded_corpus()
    hits = pole_hits(corpus, 0, edges, HarvestConfig(hits_per_band=10), torch.Generator().manual_seed(0))["high"]
    hb = hit_bands(corpus.acts[:, 0], corpus.valid, edges["high"][:, 0], "high", hits)
    assert hb.tokens == [40, 20, 5] and hb.sampled == [10, 10, 5]
    per_band = np.bincount(hb.band, weights=hb.weight, minlength=3)
    assert np.allclose(per_band, [40, 20, 5])

    rng = np.random.default_rng(0)  # two tight groups of MLP inputs, so HDBSCAN finds clusters
    centres = rng.normal(size=(2, 16))
    X = {int(t): torch.tensor(centres[i % 2] + 0.01 * rng.normal(size=16)) for i, t in enumerate(hits)}
    a = corpus.acts[hits, 0].float().numpy()
    r = pole_modes(hits, X, a, hb, ModesConfig(min_cluster_size=5, min_samples=2), seed=0)
    assert r["clusters"]
    assert np.isclose(sum(c["pole_pct"] for c in r["clusters"]) + r["other_pct"], 100)
    assert sum(sum(c["bands"]) for c in r["clusters"]) + sum(r["other_bands"]) == len(hits)
