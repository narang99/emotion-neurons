"""The emotion vectors of one run, and how they are stored.

- Everything downstream reads an `EmotionVectors`, never the recipe that built it.
- Stored as `vectors.safetensors` with these keys:
  - `<site>/<baseline>/<variant>` [emotion, dim]: baseline `mean` or `neutral`, variant `raw` or `denoised`;
  - `<site>/neutral_pcs` [k, dim], `<site>/grand_mean` [dim], `<site>/neutral_mean` [dim].
"""

from pathlib import Path

from jaxtyping import Float
from safetensors.torch import load_file, save_file
from torch import Tensor

from ..config import Baseline, Variant
from ..emotions import Site


class EmotionVectors:
    def __init__(self, tensors: dict[str, Float[Tensor, "..."]]):
        self.tensors = tensors

    def get(self, site: Site, baseline: Baseline = "mean", variant: Variant = "raw") -> Float[Tensor, "emotion dim"]:
        return self.tensors[f"{site}/{baseline}/{variant}"]

    def neutral_pcs(self, site: Site) -> Float[Tensor, "k dim"]:
        return self.tensors[f"{site}/neutral_pcs"]

    def grand_mean(self, site: Site) -> Float[Tensor, " dim"]:
        return self.tensors[f"{site}/grand_mean"]

    def neutral_mean(self, site: Site) -> Float[Tensor, " dim"]:
        return self.tensors[f"{site}/neutral_mean"]

    def mlp_write(self) -> Float[Tensor, "emotion d_model"]:
        """The MLP's emotion write, delta_e = W_down @ dA_e."""
        return self.get("mlp_out").float()

    def activation_shift(self) -> Float[Tensor, "emotion d_mlp"]:
        """dA_e: each neuron's mean activation on emotion e's train stories minus its 12-emotion mean."""
        return self.get("mlp_act").float()

    def save(self, path: Path) -> None:
        save_file({k: v.float().contiguous() for k, v in self.tensors.items()}, str(path))

    @classmethod
    def load(cls, path: Path) -> "EmotionVectors":
        return cls(load_file(str(path)))


def unit(v: Float[Tensor, "... dim"]) -> Float[Tensor, "... dim"]:
    return v / v.norm(dim=-1, keepdim=True)


def cosine_matrix(a: Float[Tensor, "m dim"], b: Float[Tensor, "n dim"]) -> Float[Tensor, "m n"]:
    return unit(a) @ unit(b).T
