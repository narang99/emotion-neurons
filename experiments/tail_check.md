# Tail check: does an emotion's write come from tail firings or from small shifts?

Date: 2026-11-19. Branch: `stack/05-causal-sets`. Tag: `exp/tail_check`. Status: kept.

Config: `configs/tail_check.yaml`. Results: `results/tail_check/tail_check.json`.

```
python scripts/tail_check.py --config configs/tail_check.yaml   # L4, 3.5 minutes
```

## Question

- The two methods assume different things.
  - Goodfire treats a neuron as a feature. Its contribution comes from rare, strong firings (the tails).
  - The emotion-vector method takes a mean over all tokens. A small shift on every token gives the same mean.
- Which one is true for the layer-25 emotion write?

## Method

- Thresholds: p3 and p97 of all 12288 layer-25 neurons, over 206,704 tokens of the general harvest corpus.
- Each token of each neuron on the 1200 train stories (tokens 50+) is in one class:
  - `high` (>= p97);
  - `low` (<= p3);
  - `middle`.
- Each emotion's shift `dA_e` splits exactly into the three classes.
  - The classes add up to Stage 2's `dA_e` within 0.3% (check in the script).
- For each class: the share of the emotion write along `delta_e` that it carries. This is computed for the emotion's sparse set, and for all neurons.

## Results

Means over the 12 emotions:

| scope | high | low | middle | total |
|---|---|---|---|---|
| sparse set: share of the write | 0.24 | 0.19 | 0.15 | 0.58 |
| sparse set: share of tokens | 0.053 | 0.051 | 0.896 | 1 |
| all neurons: share of the write | 0.35 | 0.31 | 0.34 | 1 |
| all neurons: share of tokens | 0.036 | 0.035 | 0.929 | 1 |

- **Sparse-set neurons follow the Goodfire view.**
  - About 10% of their tokens are in a tail.
  - Those tokens carry 75% of the set's write (0.44 of 0.58).
  - The set's neurons are in their tails more often on stories (5% each side) than on general text (3%).
- **The low tail matters as much as the high tail.** Tokens below p3 carry 0.19, close to the 0.24 of tokens above p97.
  - Neurons that the emotion pushes down below their normal floor are part of the write.
- **Over all neurons, the middle carries one third of the write.** 93% of tokens are in the middle, and they carry 0.34.
  - This part is small shifts spread over many neurons and tokens. The sparse set leaves most of it out.
- Per emotion, the set's tails carry 0.31 (proud) to 0.53 (loving) of the write. The middle carries 0.09 to 0.20.

## Conclusion

- Both views are partly right.
  - The neurons that the sparse search picks act like features: rare, strong firings in both directions.
  - The rest of the MLP write (about one third) is a small shift on most tokens. A feature-by-feature circuit cannot see it.

## Limits

- The thresholds come from 207k tokens of general text, not from the 2M-token harvest.
- The low tail of a SiLU neuron is a small negative value. "Below p3" thus means "pushed below its usual floor", not "strong firing".

## Causal test: clamp the tails

Script: `scripts/tail_causal.py`, config `configs/tail_causal.yaml`. Results: `results/tail_causal/tail_causal.json`.
Tag: `exp/tail_causal`.

- Method: on the 240 held-out stories, at every token of layer 25, clamp neurons into [p3, p97] of general text.
  - Only the tail firings change. Middle values stay.
  - S is the emotion's sparse set, 3 random sets of the same size, or all neurons.
  - For comparison: remove the set's mean shift `dA_e[S]` (as in `causal_sets`).
- Results: the true emotion's score, as a share of clean.

  | edit | L25 | L28 | L31 | L35 | accuracy L25 |
  |---|---|---|---|---|---|
  | clean | 1 | 1 | 1 | 1 | 0.82 |
  | clamp set tails | 0.965 | 0.976 | 0.982 | 0.986 | 0.83 |
  | clamp all neurons' tails | 0.950 | 0.968 | 0.975 | 0.982 | 0.84 |
  | remove the set's mean shift | 0.928 | 0.945 | 0.956 | 0.961 | 0.80 |
  | clamp random sets' tails | 0.997 | 0.998 | 0.998 | 0.998 | 0.82 |

- Clamping the set's tails gives about half the effect of removing its whole shift (3.5% vs 7.2% at layer 25). It is 12 times the effect of random sets.
- The decomposition gave the tails 75% of the set's write. Causally they give about 50%.
  - One reason: clamping also removes tail firings that are not about the emotion.
  - Another reason: later layers partly make up for the change.
- No accuracy or answer changes beyond 0.02. All effects are small, as in `causal_sets`.
