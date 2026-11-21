"""The target of a sparse set: one emotion's MLP write, and the activation shift behind it.

- Shift: `a[n] = abar_emotion[n] - abar_all[n]`, from story means of `mlp_act` over tokens 50+.
  - abar_emotion: the mean over the emotion's 120 stories. abar_all: the mean over all 1680 stories.
- Write: `d = W_down @ a = sum_n a[n] w_n`. Exact.
- Keeping a set S holds every other neuron at its all-story mean, so the set writes `sum_{n in S} a[n] w_n`.
"""

from jaxtyping import Bool, Float
from torch import Tensor

StoryActs = Float[Tensor, "story d_mlp"]


def emotion_shift(A: StoryActs, own: Bool[Tensor, " story"]) -> Float[Tensor, " d_mlp"]:
    return A[own].mean(0) - A.mean(0)
