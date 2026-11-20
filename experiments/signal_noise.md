# Signal, noise and offset neurons along each emotion vector

Date: 2026-11-20. Branch: `stack/06-raw-sets`. Tag: `exp/signal_noise`. Status: kept.

Config: `configs/signal_noise.yaml`. Results: `results/signal_noise/signal_noise.json`, `results/signal_noise/report.md` (modes).
Cache: `runs/signal_noise/story_classes.safetensors` (S3).

```
python scripts/signal_noise.py --config configs/signal_noise.yaml   # L4, 5 minutes (1.5 with the cache)
```

## Hypothesis

- Along an emotion direction, most neurons are unrelated. Their pushes go both ways and cancel to about zero.
- The rest carry the signal.

## Method

- Push of neuron n along the layer-25 unit vector v: `a_n (w_n . v)`, raw. No mean is removed.
- Unit of evidence: the story (mean over tokens 50+). 1680 stories.
- Per emotion, on its 100 train stories, a t-test per neuron (|t| > 4.6, about Bonferroni p < 0.01):
  - `noise`: mean push about zero;
  - `specific`: mean push not zero, and different from the other stories (11 emotions and neutral);
  - `constant`: mean push not zero, and the same on the other stories. This is an offset.
- Each push splits into tokens >= p97, <= p3 and middle (general-text thresholds).
- Held-out stories: d' of each group's summed push, own emotion vs the other stories.
- Control: a random direction.
  - The t-test does not depend on the direction: `w_n . v` is one number per neuron, and it scales the mean and the error alike.
  - So the counts are the same. Only the shares and d' change.

## Results

Means over the 12 emotions:

| group | neurons | share of the contrast (own minus other stories) | d' held-out | d', random direction | contrast from pole tokens |
|---|---|---|---|---|---|
| specific | 1559 (13%) | 0.54 | 2.38 | -0.16 | 0.73 (high 0.40, low 0.33) |
| constant | 6944 (57%) | 0.22 | 0.98 | -0.01 | 0.77 |
| noise | 3784 (31%) | 0.24 | 1.48 | 0.03 | 0.82 |
| all | 12288 | 1 | 2.13 | -0.02 | |

- **Most neurons do not average to zero.** 69% have a non-zero mean push on the emotion's stories.
  - 57% are `constant`: they push the same way on any story.
  - Their summed push is an offset. Its size and sign vary by emotion: from 0.01 (happy) to 3.6 (loving) times the total mean push.
- **The zero-mean neurons do cancel.** The noise sum is about 1% of the total push.
  - Its spread is 1.31 times that of independent noise, so the noise neurons are a little correlated.
- **The emotion-specific part is concentrated.**
  - The `specific` group separates own from other stories better than the whole MLP (d' 2.38 vs 2.13).
  - 90% of its contrast comes from 83 to 368 neurons (happy: 83).
  - Pole tokens (8% of tokens) carry 73% of its contrast.
- **The noise group still carries some contrast** (0.24, d' 1.48). These neurons are near zero on the emotion's stories but not on the others.
- Overlap with the match sets: 32% to 54% of each match set is in the `specific` group.

## Modes of the constant (offset) neurons

- Happy's strongest constant neurons fire on punctuation, function words and pronouns (`report.md`):
  - n3853: ".", " all", " they"; n1439: ":", "assistant", digits; n7555: ",", " is", " was"; n1590: " or", " if";
    n10794: ".", ",", " the"; n10947: " time", ",".
  - One exception, n10073: " smiled", " waved", " hands", " heart". It pushes the same on every story, so it is not happy-specific.
- n10947 and n10794 were the most repeated members of the per-token raw sets (`raw_sets.md`: 78% and 68% of tokens).
  - So the per-token raw sets partly picked offset neurons.

## Conclusion

- The hypothesis holds for 31% of neurons: they cancel.
- 57% of neurons add an offset that does not cancel. They fire on syntax and function words, on any text.
- The emotion-specific part is about half of the contrast. It comes from a few hundred neurons, and mostly from their pole firings.

## Limits

- Pushes are along the layer-25 unit vector, not the MLP's own write.
- The t-test treats stories as independent; stories on the same topic share words.

## Are the offset neurons at their poles? (`scripts/offset_modes.py`, CPU)

Results: `results/signal_noise/offset_modes.json`. Means over the 12 emotions, on each emotion's train stories:

| group | tokens at a pole | push from pole tokens | push from middle tokens | typical middle value on [p3, p97] |
|---|---|---|---|---|
| specific | 7.6% | 0.39 | 0.61 | 0.50 |
| constant (offset) | 6.9% | 0.23 | 0.77 | 0.50 |
| noise | 7.1% | 0.44 | 0.56 | 0.50 |

- Push shares are signed per neuron: each neuron counts toward its own net push.
- **The offset neurons are mostly in between.** 77% of their push comes from middle tokens.
  - They are at a pole on 6.9% of tokens, the general-text base rate (6%) plus a little.
  - Their middle values sit near the centre of their range. The push is a small, steady lean over most tokens.
- For comparison, the specific neurons' raw push is 39% from poles, but their emotion contrast is 73% from poles (above).
  - Their middle lean is shared by all stories; the emotion-dependent part is the pole firings.
- **The noise group's contrast (0.24) comes from weak but real differences.**
  - 89% of it comes from noise neurons with 2 < |t| < 4.6 for own vs other stories (1400 to 2150 neurons per emotion).
  - These fail the strict cut one by one, but add up.
