"""Sparse sets: the fewest neurons of one MLP whose measured shifts rebuild an emotion's write.

- gaps: the target `d = W_down a`, with `a` the emotion's mean activation minus the all-story mean.
- greedy: add the neuron that rebuilds the most of what is still missing, until 90% of d is rebuilt.
"""

from .gaps import emotion_shift
from .greedy import GreedyRun, first_reaching, greedy_set

__all__ = ["GreedyRun", "emotion_shift", "first_reaching", "greedy_set"]
