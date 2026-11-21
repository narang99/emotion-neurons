import torch

from emotion_neurons.config import VectorsConfig
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.texts import StorySplit
from emotion_neurons.vectors import mean_contrast
from emotion_neurons.vectors.diff_means import diff_of_means

torch.manual_seed(0)


def toy(per: int = 4, neutral: int = 8, dim: int = 6) -> tuple[torch.Tensor, list[str], StorySplit]:
    labels = [e for e in EMOTIONS for _ in range(per)] + ["neutral"] * neutral
    split = StorySplit(
        label=torch.tensor([EMOTIONS.index(x) if x != "neutral" else -1 for x in labels]),
        heldout=torch.zeros(len(labels), dtype=torch.bool),
        topic=torch.zeros(len(labels), dtype=torch.long),
    )
    return torch.randn(len(labels), dim, dtype=torch.float64), labels, split


def test_all_baseline_reproduces_the_emotion_vectors():
    X, labels, split = toy()
    vectors, _ = mean_contrast({"resid_post": X}, split, VectorsConfig())
    for e in ("happy", "afraid"):
        got = diff_of_means(X, labels, list(EMOTIONS), e, baseline="all")
        torch.testing.assert_close(got, vectors.get("resid_post", "mean", "raw")[EMOTIONS.index(e)])


def test_denoised_emotion_vectors_with_the_neutral_reference():
    X, labels, split = toy()
    vectors, _ = mean_contrast({"resid_post": X}, split, VectorsConfig())
    got = diff_of_means(X, labels, list(EMOTIONS), "sad", baseline="all", pca_reference="neutral", pca_variance=0.5)
    torch.testing.assert_close(got, vectors.get("resid_post", "mean", "denoised")[EMOTIONS.index("sad")])


def test_two_categories_all_is_half_of_rest():
    X = torch.randn(10, 4, dtype=torch.float64)
    labels = ["harmful"] * 5 + ["harmless"] * 5
    cats = ["harmful", "harmless"]
    rest = diff_of_means(X, labels, cats, "harmful", baseline="rest")
    torch.testing.assert_close(diff_of_means(X, labels, cats, "harmful", baseline="all"), rest / 2)
    torch.testing.assert_close(rest, X[:5].mean(0) - X[5:].mean(0))


def test_keep_restricts_the_examples():
    X = torch.randn(8, 3, dtype=torch.float64)
    labels = ["a", "a", "a", "a", "b", "b", "b", "b"]
    keep = torch.tensor([True, True, False, False, True, True, False, False])
    got = diff_of_means(X, labels, ["a", "b"], "a", keep=keep)
    torch.testing.assert_close(got, X[:2].mean(0) - X[4:6].mean(0))
