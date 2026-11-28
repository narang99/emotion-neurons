"""How emotion vectors are built from per-story means.

- A recipe takes the story means of every site and the split, and returns an `EmotionVectors`.
- To try another procedure, add a recipe here and name it in `vectors.recipe`.
  - Nothing downstream changes, since it only reads `EmotionVectors`.
"""

from typing import Protocol

from jaxtyping import Bool, Float
from pydantic import BaseModel
from torch import Tensor

from ..capture import StoryMeans
from ..config import VectorsConfig
from ..emotions import EMOTIONS, Site
from ..texts import StorySplit
from .denoise import neutral_pcs, remove
from .diff_means import category_means, diff_of_means
from .emotion_vectors import EmotionVectors


class DenoiseInfo(BaseModel):
    k: int
    variance_explained: float


class VectorRecipe(Protocol):
    def __call__(
        self, means: StoryMeans, split: StorySplit, cfg: VectorsConfig
    ) -> tuple[EmotionVectors, dict[Site, DenoiseInfo]]: ...


def emotion_means(
    x: Float[Tensor, "story dim"], split: StorySplit, among: Bool[Tensor, " story"]
) -> Float[Tensor, "emotion dim"]:
    return category_means(x, split.label, len(EMOTIONS), among)


def mean_contrast(means: StoryMeans, split: StorySplit, cfg: VectorsConfig) -> tuple[EmotionVectors, dict[Site, DenoiseInfo]]:
    """The emotions paper's recipe, on the train stories.

    - `mean` baseline: emotion mean minus the mean over the 12 emotions.
    - `neutral` baseline: emotion mean minus the neutral-story mean.
    - `denoised` variants have the neutral PCs removed.
    """
    tensors: dict[str, Tensor] = {}
    info: dict[Site, DenoiseInfo] = {}
    neutral_train = split.train & split.neutral
    for site, x in means.items():
        mu = emotion_means(x, split, split.train)
        grand, neutral = mu.mean(0), x[neutral_train].mean(0)
        pcs, explained = neutral_pcs(x[neutral_train], cfg.denoise_variance)
        for baseline, v in (("mean", diff_of_means(mu, "all")), ("neutral", mu - neutral)):
            tensors[f"{site}/{baseline}/raw"] = v
            tensors[f"{site}/{baseline}/denoised"] = remove(v, pcs)
        tensors[f"{site}/neutral_pcs"] = pcs
        tensors[f"{site}/grand_mean"] = grand
        tensors[f"{site}/neutral_mean"] = neutral
        info[site] = DenoiseInfo(k=pcs.shape[0], variance_explained=explained)
    return EmotionVectors(tensors), info


RECIPES: dict[str, VectorRecipe] = {"mean_contrast": mean_contrast}
