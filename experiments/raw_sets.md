# Raw per-token sets: is the emotion direction a sum of neurons at their poles?

Date: 2026-11-20. Branch: `stack/06-raw-sets`. Tag: `exp/raw_sets`. Status: kept.

Config: `configs/raw_sets.yaml`. Results: `results/raw_sets/raw_sets.json`. Orders: `runs/raw_sets/orders.safetensors` (S3).

```
python scripts/raw_sets.py --config configs/raw_sets.yaml   # L4, 2.5 minutes
```

## Question

- At a token where an emotion is strong, can a small set of neurons, with their raw activations, rebuild the emotion direction?
- If yes: are those neurons at their poles (<= p3 or >= p97 of general text)?
- No mean is removed anywhere.

## Method

- Tokens: per emotion, the 200 held-out tokens (50+) of its own stories with the highest layer-25 score. At most 20 per story.
  - Score: `(resid_post - centre) . unit_e`.
- Terms: `u_n(t) = a_n(t) w_n`, with `w_n` the column of `W_down`.
- Greedy (`decomposition.raw_greedy`): add the neuron that raises `cos(sum_S u_n(t), v_e)` most. 1500 steps.
  - `v_e` is the layer-25 emotion unit vector. It is the target only.
  - Sets are read at the first size that reaches cos 0.5, 0.7 and 0.9.
- Each member's activation is placed among its own general-corpus quantiles (207k tokens).
- Controls at the same tokens:
  - `random`: the same greedy toward a random direction;
  - `largest`: the k neurons with the largest terms `|a_n| |w_n|`;
  - `all`: every neuron.

## Results

- **The MLP output at these tokens points only a little along the emotion vector.**
  - Cosine of the whole MLP-25 output with `v_e`: 0.02 to 0.11 (random direction: 0.00).
- **Greedy can build a random direction about as easily.**

  | target | median k at cos 0.5 | reached 0.7 | median k at cos 0.7 | best cos |
  |---|---|---|---|---|
  | emotion vector | 206 to 474 (mean 277) | 58% of tokens | 1027 to 1190 | 0.65 to 0.73 |
  | random direction | 268 to 300 (mean 285) | 36% of tokens | 1086 to 1264 | 0.68 to 0.70 |

  - 12288 terms in 4096 dimensions can be combined toward almost any direction.
  - So a per-token set is not unique, and "a set exists" is not evidence by itself.
  - The emotion vector needs slightly fewer neurons, and reaches 0.7 more often.
- **The set members are mostly not at their poles.** At cos 0.5, means over the 12 emotions:

  | group | <p3 | p3-p25 | p25-p75 | p75-p97 | >=p97 |
  |---|---|---|---|---|---|
  | emotion sets | 0.049 | 0.307 | 0.277 | 0.315 | 0.052 |
  | random-direction sets | 0.001 | 0.139 | 0.718 | 0.140 | 0.001 |
  | largest terms (same k) | 0.403 | 0.088 | 0.009 | 0.095 | 0.404 |
  | all neurons | 0.036 | 0.227 | 0.472 | 0.229 | 0.036 |

  - Pole members: 10% of the set. All neurons: 7%. Random-direction sets: 0.2%.
  - Pole members carry 23% of the set's projection on `v_e` (high 0.12, low 0.11). In random-direction sets: 1%.
  - At cos 0.7 the pole share drops to 1 to 3%. The extra members are fine adjustments near the median.
- **What separates emotion sets from random-direction sets is moderate displacement, not poles.**
  - p75-p97: 0.315 vs 0.140. p3-p25: 0.307 vs 0.139.
  - The largest terms are 81% at a pole, but greedy does not pick them. They write mostly off the emotion direction.
- **Some neurons repeat across tokens.**
  - happy, cos 0.5: n9935 in 85% of the token sets, n10947 78%, n6882 76%, n48 74%. 6 of the top 10 are in the match set.
  - Members are in the match set 2 times as often as chance (0.10 to 0.23 vs 0.02 to 0.11).
  - n48 is one of the "a feeling was just named" neurons (`experiments/stage3_l25.md`).

## Conclusion

- At this layer, the emotion direction is not a sum of a few neurons at their poles.
  - The neurons at their poles write large terms, mostly off the emotion direction.
  - The neurons that point the output toward the direction are mostly between p3 and p97, with many near p10-p25 and p75-p97.
- The per-token fit is weak evidence in either direction: the same greedy fits a random direction with 3% more neurons.

## Limits

- Cosine lets small terms cancel the off-target parts of large terms. This favours middle activations as fine adjustments.
- One layer (25), one target (the layer-25 unit vector), 200 tokens per emotion.
- Open (README): what to do with neurons near p75; whether "a feeling was just named" neurons carry the emotion-specific part.

## Pole-only candidates

Script: `scripts/raw_pools.py`, config `configs/raw_pools.yaml`. Results: `results/raw_pools/raw_pools.json`. Tag: `exp/raw_pools`.

- Question: if greedy may use only the neurons at a pole at that token, how close does it get to the emotion vector?
  - This checks whether the search method caused the result above.
- Pools per token: `all`; `poles` (<= p3 or >= p97, about 860 neurons); `middle`; `middle_sized` (a random middle subset with as many neurons as `poles`).
- Same 200 tokens per emotion. Targets: the emotion vector, and a random direction.
- Results, means over the 12 emotions (happy in brackets):

  | pool | neurons | cos of the whole pool's sum | best greedy cos | k at best | best cos, random direction |
  |---|---|---|---|---|---|
  | all | 12288 | 0.08 (0.08) | 0.69 (0.66) | 1500 (step cap) | 0.68 |
  | poles | 855 (855) | 0.07 (0.09) | **0.32 (0.33)** | 259 (248) | 0.27 |
  | middle | 11400 | 0.03 (0.02) | 0.69 (0.68) | 1500 (step cap) | 0.68 |
  | middle_sized | 855 | 0.01 (0.01) | 0.25 (0.23) | 282 (270) | 0.24 |

- **The pole neurons cannot build the emotion vector.** The best set from them reaches cos 0.32. All neurons reach 0.69.
- **The poles lean toward the emotion a little more than chance.**
  - Best cos: 0.32 toward the emotion, 0.27 toward a random direction. The same-size middle pool: 0.25 and 0.24.
  - Sum of the whole pool: 0.07 for poles, 0.01 for the same-size middle pool.
- In every one of the 12 emotions, poles beat the same-size middle pool and beat their own random-direction control.
- So the result above does not come from the search method. With only pole neurons, the emotion direction is out of reach.
- Limit: `all` and `middle` hit the 1500-step cap, so their best cosines are lower bounds.
