"""Sparse sets: the fewest neurons of one MLP whose measured shifts rebuild an emotion's gap write.

- gaps: the target `d = W_down a` and the story groups behind it.
- greedy: 0/1 greedy on the leftover (the earlier method, the baseline).
- relaxed: a mask in [0, 1] with an L1 penalty, solved convexly (this method).
- crossval: where to stop, by held-out error over 6 folds.
- scoring: leftover on the fitting stories, random sets.
"""

from .crossval import fold_split, no_heldout, prefix_errors, set_error
from .gaps import EmotionGroups, activation_gap, emotion_groups
from .greedy import GreedyRun, first_below, residual_greedy
from .relaxed import MaskSolution, lambda_max, lambda_path, lipschitz, scaled_columns, solve_mask, support
from .scoring import random_members, set_leftover

__all__ = [
    "EmotionGroups",
    "GreedyRun",
    "MaskSolution",
    "activation_gap",
    "emotion_groups",
    "first_below",
    "fold_split",
    "lambda_max",
    "lambda_path",
    "lipschitz",
    "no_heldout",
    "prefix_errors",
    "random_members",
    "residual_greedy",
    "scaled_columns",
    "set_error",
    "set_leftover",
    "solve_mask",
    "support",
]
