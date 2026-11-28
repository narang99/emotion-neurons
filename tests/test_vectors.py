import torch

from emotion_neurons.capture import pooled_sums
from emotion_neurons.config import VectorsConfig
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.texts import StorySplit
from emotion_neurons.vectors import mean_contrast, neutral_pcs, remove

torch.manual_seed(0)


def toy_split(per_emotion: int = 4, neutral: int = 8) -> StorySplit:
    labels = torch.arange(len(EMOTIONS)).repeat_interleave(per_emotion).tolist() + [-1] * neutral
    return StorySplit(
        label=torch.tensor(labels),
        heldout=torch.zeros(len(labels), dtype=torch.bool),
        topic=torch.zeros(len(labels), dtype=torch.long),
    )


def test_mean_baseline_vectors_sum_to_zero_over_emotions():
    split = toy_split()
    x = torch.randn(len(split.label), 6, dtype=torch.float64)
    vectors, _ = mean_contrast({"resid_post": x}, split, VectorsConfig())
    torch.testing.assert_close(vectors.get("resid_post", "mean", "raw").sum(0), torch.zeros(6, dtype=torch.float64))


def test_denoised_vectors_are_orthogonal_to_the_neutral_pcs():
    neutral = torch.randn(20, 6, dtype=torch.float64)
    pcs, _ = neutral_pcs(neutral, 0.5)
    v = remove(torch.randn(3, 6, dtype=torch.float64), pcs)
    torch.testing.assert_close(v @ pcs.T, torch.zeros(3, pcs.shape[0], dtype=torch.float64))


def test_pooled_sums_split_tokens_at_the_split_token():
    acts = {site: torch.ones(2, 5, 3) for site in ("resid_mid", "resid_post", "mlp_act")}
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    sums, counts = pooled_sums(acts, mask, split_token=2)
    assert counts.tolist() == [[2, 3], [2, 1]]
    assert sums["mlp_act/late"][:, 0].tolist() == [3.0, 1.0]
