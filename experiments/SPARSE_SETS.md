# Sparse neuron sets: the methods we tried

The question: what is the smallest set of layer-25 MLP neurons that builds an emotion's direction?

- The code is not on this branch. Each method's code, config, results and full log are at its git tag.
  - Read one: `git show <tag>:experiments/<log>.md`. Run one: `git checkout <tag>`.
- Each method has its setup, its score, where it stops, its result and its tag.

## Notation

- `a_n(t)`: activation of neuron n at token t (the input to `down_proj`). `w_n`: column n of `W_down`.
- MLP output at t: `sum_n a_n(t) w_n`. Exact.
- `abar_X[n]`: the mean of `a_n` over tokens 50+ of each story, then over the stories of group X.
- An emotion's MLP write against a baseline B: `d = sum_n u_n`, with `u_n = (abar_own[n] - abar_B[n]) w_n`. Exact.
- Mean ablation of a set S: S keeps its emotion values, every other neuron sits at its baseline mean.
  - The write is then `sum_{n in S} u_n`.
- Poles: `a_n <= p3` or `a_n >= p97` of the neuron's own distribution on general text (`results/poles/`).
- Greedy search: start from an empty set, add the neuron that improves the score most, and repeat.
  - The Gram matrix `(W^T W) * (shift shift^T)` makes each step exact and fast.

## The methods, in order

1. **Dims, then neurons** (tag `exp/stage3_l25`, log `stage3_l25.md`). Failed.
   - Keep the residual dims that carry 90% of the write, then the neurons that carry 90% of each dim.
   - Result: the rule keeps all 12288 neurons.

2. **Greedy on cosine, stop at a train cosine** (tag `exp/min_set_l25`, log `top64_l25.md` at tag `exp/sparse_l25`). Superseded by 3.
   - Score: `cos(sum_S u_n, d)`. Baseline: the 12-emotion mean.
   - Result: cosine 0.9 needs 700 to 1050 neurons.
   - Problem: a train cosine says nothing about held-out stories.

3. **Match sets: greedy on cosine, held-out stop** (tag `exp/sparse_l25`, log `top64_l25.md`). Kept. This is method 1.
   - Score: `cos(sum_S u_n, d)`. Baseline: the 12-emotion mean.
   - The neutral directions (top PCs of neutral stories, 50% variance) are removed from `W`.
   - Stop: the set's held-out cosine reaches the whole MLP's held-out cosine.
   - Result: 268 to 1401 neurons (mean 728, about 6%).
   - Variants, none better by more than 6.5%:
     - 1-swaps and shrinking: 0 to 3% fewer neurons. Greedy is already a 1-swap local optimum.
     - Cross-fit on two halves of the stories.
     - A shrunk target `max(0, 1 - kappa / t^2)` that leans on reproducible shifts.
   - Cosine rises about linearly in log(set size). No knee, no small core.

4. **The same at every layer** (tag `exp/sparse_layers`, log `sparse_layers.md`). Kept.
   - Cosine 0.9 needs about 1200 neurons at layers 10 to 20, 570 at layers 28 to 34, and 38 at layer 35.

5. **Whole-network circuits for the emotion score** (tags `exp/circuits_l25-round1`, `-round2`, log `circuits_l25.md` at the round-2 tag). Failed.
   - Rank neurons of all layers by their effect on the layer-25 score, against zero, then against the mean.
   - Result: faithfulness about 0 or negative at 100 to 3000 neurons.

6. **Per token, greedy on raw `a_n(t) w_n`** (tag `exp/raw_sets`, log `raw_sets.md`). Superseded by 8.
   - No baseline. Target: the L25 emotion vector, at tokens with a high emotion score.
   - Result: a random direction is fit with about the same set size (285 vs 277). Per-token sets are not unique.

7. **The same, with only pole neurons as candidates** (tag `exp/raw_pools`, log `raw_sets.md`). Superseded by 8.
   - Result: poles alone reach cosine 0.32 (all neurons 0.69). The direction is not buildable from pole neurons alone.

8. **Signal, offset or noise: a t-test per neuron** (tag `exp/signal_noise`, log `signal_noise.md`). Kept.
   - Per neuron, the story-level push `abar_n (w_n . v)` along the emotion vector v.
   - A t-test with `|t| > 4.6` (about Bonferroni p < 0.01) sorts each neuron into one group:
     - noise: mean push about 0;
     - constant offset: the same push for every emotion;
     - emotion-specific: a different push for its own emotion.
   - Result: 31% noise, 57% constant offset (syntax neurons, on any text), 13% specific.
     - The specific 13% carry 54% of the contrast. 90% of it comes from 83 to 368 neurons, 73% from pole tokens.
   - The test does not depend on the direction, because `w_n . v` is one number per neuron.

9. **Signal sets: greedy on the leftover** (tag `exp/residual_sets`, log `residual_sets.md`). Kept. This is method 2.
   - Score: the leftover `|d - sum_S u_n| / |d|`, which checks direction and length.
   - Gain of candidate j: `2 (c_j - t_j) - diag_j`, with `c = shift * (W^T d)`, `diag = shift^2 |w|^2`,
     and `t` the current set's term.
   - Baseline: all other stories (11 emotions and neutral). `W` is not denoised.
   - Stop: the leftover reaches the train target's own sampling noise (0.24 to 0.37).
   - Result: 1499 to 2232 neurons (12 to 18%). A random set of the same size leaves 0.88 to 0.93.
     - The rest cancel as a vector (1.05 to 1.38 times the noise floor, 6 folds).
     - But the rest still adds 11 to 23% along the emotion axis (z 20 to 27).
   - The match set (3) is 78 to 92% inside the signal set.

## What the methods agree on

- Methods 3 and 9 trace the same curve of cosine vs set size. They differ only in where they stop.
- The write is spread over about 2000 neurons. There is no small core.
- About 70% of the emotion-specific write comes from pole firings, on about 8% of tokens.
- Mean ablation is needed. Zero ablation drops the cosine to 0 to 0.3.
  - The mean write of the other neurons is 4 to 7 times the emotion gap.
- Blind Haiku judges (tag `exp/judge_sets`, log `judge_sets.md`) find about 1 in 5 set neurons emotion-related.
  - Only in the signal set: 0.21. In both sets: 0.21. Random: 0.00. About 37% look like syntax.
