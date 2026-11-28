"""Difference of means over categories of examples, shared by every method that builds a direction.

- Category means are taken first, so categories count equally whatever their sizes.
- A direction is one category's mean minus a baseline:
  - `all`: the mean of all category means, its own included (the emotion vectors);
  - `rest`: the mean of the other categories' means (the refusal direction, with two categories).
  - `all` is `(n - 1) / n` times `rest`, so the two differ in length only.
"""

from typing import Literal

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

Against = Literal["all", "rest"]
CategoryMeans = Float[Tensor, "category ..."]


def category_means(
    x: Float[Tensor, "example dim"], label: Int[Tensor, " example"], n_categories: int, among: Bool[Tensor, " example"]
) -> Float[Tensor, "category dim"]:
    """`among` limits the examples used, e.g. to the train split. Examples with a label outside the range are ignored."""
    masks = [among & (label == c) for c in range(n_categories)]
    assert all(mask.any() for mask in masks), "a category has no examples"
    return torch.stack([x[mask].mean(0) for mask in masks])


def mean_of_others(means: CategoryMeans, left_out: int) -> Tensor:
    others = [i for i in range(means.shape[0]) if i != left_out]
    return means[others].mean(0)


def diff_of_means(means: CategoryMeans, against: Against) -> CategoryMeans:
    """The direction of every category at once, in the order of `means`."""
    n = means.shape[0]
    assert n >= 2, "a difference of means needs at least two categories"
    if against == "all":
        return means - means.mean(0)
    return means - torch.stack([mean_of_others(means, i) for i in range(n)])
