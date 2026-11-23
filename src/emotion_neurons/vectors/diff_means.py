"""Generic difference of means: a target category's mean minus a baseline of category means, optionally denoised.

- Category means are taken first, so categories count equally whatever their sizes.
- Baseline:
  - "all": the mean of all category means, the target's included (the emotion vectors' `mean` baseline);
  - "rest": the mean of the other categories' means.
  - With two categories, "all" gives exactly half of "rest": the same direction (the refusal case).
- Denoising (optional): the top principal components of a reference category's examples, enough to explain
  `variance` of it, are projected out (the emotion vectors' neutral-story PCs). The refusal paper does not denoise.
"""

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from .denoise import neutral_pcs, remove

Examples = Float[Tensor, "example dim"]


def category_means(
    X: Examples, labels: list[str], categories: list[str], keep: Bool[Tensor, " example"] | None = None
) -> Float[Tensor, "category dim"]:
    keep = torch.ones(len(labels), dtype=torch.bool) if keep is None else keep
    rows = []
    for c in categories:
        mask = torch.tensor([label == c for label in labels]) & keep
        assert mask.any(), f"no examples of category {c!r}"
        rows.append(X[mask].mean(0))
    return torch.stack(rows)


def diff_of_means(
    X: Examples,
    labels: list[str],
    categories: list[str],
    target: str,
    baseline: str = "rest",
    pca_reference: str | None = None,
    pca_variance: float = 0.5,
    keep: Bool[Tensor, " example"] | None = None,
) -> Float[Tensor, " dim"]:
    """`keep` limits the examples used (e.g. the train split)."""
    means = category_means(X, labels, categories, keep)
    t = categories.index(target)
    if baseline == "all":
        base = means.mean(0)
    elif baseline == "rest":
        base = means[[i for i in range(len(categories)) if i != t]].mean(0)
    else:
        raise ValueError(f"baseline must be 'all' or 'rest', not {baseline!r}")
    direction = means[t] - base
    if pca_reference is not None:
        keep_ref = torch.tensor([label == pca_reference for label in labels]) & (keep if keep is not None else True)
        pcs, _ = neutral_pcs(X[keep_ref], pca_variance)
        direction = remove(direction[None], pcs)[0]
    return direction
