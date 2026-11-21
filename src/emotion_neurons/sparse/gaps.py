"""The target of a sparse set: one emotion's gap write, and the activation shift behind it.

- Story groups for emotion e: own = e's stories, other = the 11 other emotions and neutral.
- Shift: `a[n] = abar_own[n] - abar_other[n]`, from story means of `mlp_act` over tokens 50+.
- Gap write: `d = W_down @ a = sum_n a[n] w_n`. Exact.
- Keeping a set S holds every other neuron at the other-story mean, so the write becomes `sum_{n in S} a[n] w_n`.
"""

from jaxtyping import Bool, Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from ..emotions import EMOTIONS
from ..texts import StorySplit

StoryActs = Float[Tensor, "story d_mlp"]


class EmotionGroups(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    own: Bool[Tensor, " story"]
    other: Bool[Tensor, " story"]
    own_heldout: Bool[Tensor, " story"]
    other_heldout: Bool[Tensor, " story"]


def emotion_groups(split: StorySplit, emotion: str) -> EmotionGroups:
    is_own = split.label == EMOTIONS.index(emotion)
    return EmotionGroups(
        own=is_own & split.train,
        other=~is_own & split.train,
        own_heldout=is_own & split.heldout,
        other_heldout=~is_own & split.heldout,
    )


def activation_gap(A: StoryActs, own: Bool[Tensor, " story"], other: Bool[Tensor, " story"]) -> Float[Tensor, " d_mlp"]:
    return A[own].mean(0) - A[other].mean(0)
