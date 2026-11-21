# Relaxed mask and fold stops for sparse sets (superseded)

Date: 2026-11-21. Branch: `stack/relaxed-mask`. Tag: `exp/sparse_relaxed_l25`. Status: superseded by `exp/greedy_sets_l25`.

## Question

- Does a convex search find a smaller set of layer-25 neurons than greedy?
- Is there a simpler, defensible place to stop a set?

## Method

- Target: the MLP write of an emotion's stories minus the other stories (`sparse/gaps.py`). Neurons outside the set sit at the other-story mean.
- Relaxed mask (`sparse/relaxed.py`): each neuron keeps its measured shift; only a mask in [0, 1] is solved, with an L1 penalty.
  - The set is every neuron with a mask above 0.
- Greedy (`sparse/greedy.py`): the method of `exp/residual_sets`, as the baseline.
- Two stop rules:
  - the target's own sampling noise (about 90% of the write rebuilt);
  - 6-fold held-out early stopping (`sparse/crossval.py`).

## Results

- The relaxed mask needs more neurons than greedy for the same share of the write.
  - happy: 1787 vs 1501 neurons (90% rebuilt). 75% of the relaxed set is in the greedy set.
  - inspired: 2678 vs 2220 neurons (92% rebuilt). 70% in the greedy set.
  - Most relaxed masks are partial (1302 of happy's 1787). Counting each at its full shift wastes neurons.
- Held-out early stopping has no stop point.
  - On new stories, the prediction keeps getting better up to all 12288 neurons (happy: error 0.758 at 713 neurons, 0.711 at 3000, 0.707 with all).
- The greedy baseline reproduces `exp/residual_sets`: happy 1501 neurons vs 1499.

## Why superseded

- Neither part makes the sets smaller or the stop simpler.
- The user chose the simplest version: greedy, all stories, a fixed 90% of the write (`exp/greedy_sets_l25`).
