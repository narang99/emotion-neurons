"""Per-story pooled sums of hooked sites, and the per-story means built from them.

- A run keeps sums, not activations: averages, differences and the MLP's write are all linear in them.
- The capture step of `extraction.emotion_vectors` writes `<site>/<early|late>` sums and `counts` [story, 2].
  - late: the example set's rendered positions; early: the other real tokens.
"""

from pathlib import Path

import torch
from einops import einsum
from jaxtyping import Bool, Float, Int
from safetensors.torch import load_file
from torch import Tensor

from ..emotions import HOOKED_SITES, RANGES, Site

StoryMeans = dict[Site, Float[Tensor, "story dim"]]


def range_masks(
    attention_mask: Int[Tensor, "batch seq"], late: Bool[Tensor, "batch seq"]
) -> dict[str, Bool[Tensor, "batch seq"]]:
    real = attention_mask.bool()
    return {"early": real & ~late, "late": real & late}


def split_masks(attention_mask: Int[Tensor, "batch seq"], split_token: int) -> dict[str, Bool[Tensor, "batch seq"]]:
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    return range_masks(attention_mask, (positions >= split_token).expand_as(attention_mask))


def pooled_sums(
    acts: dict[Site, Float[Tensor, "batch seq dim"]], attention_mask: Int[Tensor, "batch seq"], split_token: int
) -> tuple[dict[str, Float[Tensor, "batch dim"]], Int[Tensor, "batch 2"]]:
    """fp32 sums per site and range, and the token count per range."""
    return pooled_range_sums(acts, split_masks(attention_mask, split_token))


def pooled_position_sums(
    acts: dict[Site, Float[Tensor, "batch seq dim"]], attention_mask: Int[Tensor, "batch seq"], late: Bool[Tensor, "batch seq"]
) -> tuple[dict[str, Float[Tensor, "batch dim"]], Int[Tensor, "batch 2"]]:
    """As `pooled_sums`, with the late range given per row (an example set's `Rendered.positions`)."""
    return pooled_range_sums(acts, range_masks(attention_mask, late))


def pooled_range_sums(
    acts: dict[Site, Float[Tensor, "batch seq dim"]], masks: dict[str, Bool[Tensor, "batch seq"]]
) -> tuple[dict[str, Float[Tensor, "batch dim"]], Int[Tensor, "batch 2"]]:
    counts = torch.stack([masks[r].sum(1) for r in RANGES], 1)
    return sums_per_site_and_range(acts, masks), counts


def sums_per_site_and_range(
    acts: dict[Site, Float[Tensor, "batch seq dim"]], masks: dict[str, Bool[Tensor, "batch seq"]]
) -> dict[str, Float[Tensor, "batch dim"]]:
    sums = {}
    for site in HOOKED_SITES:
        for name, mask in masks.items():
            sums[f"{site}/{name}"] = einsum(acts[site].float(), mask.float(), "batch seq dim, batch seq -> batch dim")
    return sums


def late_means(run_dir: Path) -> StoryMeans:
    """Mean over each story's late tokens, in fp64. mlp_out = resid_post - resid_mid, exact."""
    sums = load_file(str(run_dir / "sums.safetensors"))
    late = sums["counts"][:, 1:2].double()
    means: StoryMeans = {site: sums[f"{site}/late"].double() / late for site in HOOKED_SITES}
    means["mlp_out"] = means["resid_post"] - means["resid_mid"]
    return means
