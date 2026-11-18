"""Per-story pooled sums of hooked sites, and the per-story means built from them.

- A run keeps sums, not activations: averages, differences and the MLP's write are all linear in them.
- Two files exist:
  - single layer (`extract.py`): `<site>/<early|late>` sums and `counts` [story, 2];
  - all layers (`extract_layers.py`): `{layer}/<site>` late sums, `embed`, and `late_count` [story].
"""

from pathlib import Path

import torch
from einops import einsum
from jaxtyping import Bool, Float, Int
from safetensors.torch import load_file
from torch import Tensor

from ..emotions import HOOKED_SITES, RANGES, Site

StoryMeans = dict[Site, Float[Tensor, "story dim"]]


def split_masks(attention_mask: Int[Tensor, "batch seq"], split_token: int) -> dict[str, Bool[Tensor, "batch seq"]]:
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    real = attention_mask.bool()
    return {"early": real & (positions < split_token), "late": real & (positions >= split_token)}


def pooled_sums(
    acts: dict[Site, Float[Tensor, "batch seq dim"]], attention_mask: Int[Tensor, "batch seq"], split_token: int
) -> tuple[dict[str, Float[Tensor, "batch dim"]], Int[Tensor, "batch 2"]]:
    """fp32 sums per site and range, and the token count per range."""
    masks = split_masks(attention_mask, split_token)
    sums = {
        f"{site}/{name}": einsum(acts[site].float(), mask.float(), "batch seq dim, batch seq -> batch dim")
        for site in HOOKED_SITES
        for name, mask in masks.items()
    }
    counts = torch.stack([masks[r].sum(1) for r in RANGES], 1)
    return sums, counts


def late_means(run_dir: Path) -> StoryMeans:
    """Single-layer run: mean over each story's late tokens, in fp64. mlp_out = resid_post - resid_mid, exact."""
    sums = load_file(str(run_dir / "sums.safetensors"))
    late = sums["counts"][:, 1:2].double()
    means: StoryMeans = {site: sums[f"{site}/late"].double() / late for site in HOOKED_SITES}
    means["mlp_out"] = means["resid_post"] - means["resid_mid"]
    return means


class LayerMeans:
    """All-layer run: per-story late means of every layer's sites, loaded once."""

    def __init__(self, run_dir: Path):
        self.sums = load_file(str(run_dir / "sums.safetensors"))
        self.count = self.sums.pop("late_count").double()[:, None]
        self.n_layers = 1 + max(int(k.split("/")[0]) for k in self.sums if k != "embed")

    def site(self, layer: int, site: Site) -> Float[Tensor, "story dim"]:
        if site == "mlp_out":
            return self.site(layer, "resid_post") - self.site(layer, "resid_mid")
        return self.sums[f"{layer}/{site}"].double() / self.count

    def embed(self) -> Float[Tensor, "story d_model"]:
        return self.sums["embed"].double() / self.count

    def attention_write(self, layer: int) -> Float[Tensor, "story d_model"]:
        before = self.embed() if layer == 0 else self.site(layer - 1, "resid_post")
        return self.site(layer, "resid_mid") - before

    def layer_means(self, layer: int) -> StoryMeans:
        return {site: self.site(layer, site) for site in ("resid_mid", "resid_post", "mlp_act", "mlp_out")}
