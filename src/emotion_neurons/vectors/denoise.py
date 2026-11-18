"""Neutral-PC denoising: remove the directions that vary most across neutral stories.

- Those directions move a lot on text with no emotion, so a vector's component along them is treated as noise.
- The projection is linear and fixed per site, so it commutes with sums and differences of means.
"""

import torch
from jaxtyping import Float
from torch import Tensor


def neutral_pcs(neutral: Float[Tensor, "story dim"], variance_share: float) -> tuple[Float[Tensor, "k dim"], float]:
    """The top principal directions of the neutral stories explaining `variance_share` of their variance."""
    centred = neutral - neutral.mean(0)
    _, singular, directions = torch.linalg.svd(centred, full_matrices=False)
    variance = singular**2 / (singular**2).sum()
    k = int((variance.cumsum(0) < variance_share).sum().item()) + 1
    return directions[:k], variance[:k].sum().item()


def remove(v: Float[Tensor, "... dim"], pcs: Float[Tensor, "k dim"]) -> Float[Tensor, "... dim"]:
    return v - (v @ pcs.T) @ pcs
