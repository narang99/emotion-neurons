# Greedy sets: the neurons that build each emotion's write

Date: 2026-11-21. Branch: `stack/greedy-sets`. Tag: `exp/greedy_sets_l25`. Status: kept.

Config: `configs/greedy_sets_l25.yaml`. Results: `results/greedy_sets_l25/`.

```
python scripts/greedy_sets.py --config configs/greedy_sets_l25.yaml   # CPU, 9 s
```

## Question

- Which layer-25 neurons build each emotion's MLP write? Give the list, for analysis of each neuron.

## Method

- Write: `W_down @ (mean activation on the emotion's 120 stories - mean on all 1680 stories)`.
  - Story means over tokens 50+. All stories, no held-out split, no PCA.
  - The write is exactly the sum of 12288 neuron terms.
- Neurons outside the set sit at their all-story mean.
- Greedy: add the neuron that rebuilds the most of what is still missing.
- Stop: the set rebuilds 90% of the write.
  - The simplest rule, the same for every emotion. Chosen over a noise-level stop and fold stops (`exp/sparse_relaxed_l25`).

## Results

| emotion | 50% | 75% | 90% (the set) | 95% | greedy stops by itself at |
|---|---|---|---|---|---|
| happy | 87 | 533 | 1527 | 2495 | 6786 (98.52%) |
| inspired | 207 | 775 | 1887 | 2964 | 6935 (98.01%) |
| loving | 94 | 511 | 1472 | 2371 | 6633 (98.64%) |
| proud | 259 | 850 | 1931 | 3039 | 6944 (97.93%) |
| calm | 261 | 883 | 2025 | 3151 | 6707 (97.78%) |
| desperate | 209 | 759 | 1789 | 2802 | 6833 (98.18%) |
| angry | 164 | 714 | 1759 | 2755 | 6897 (98.24%) |
| guilty | 174 | 732 | 1754 | 2778 | 6698 (98.27%) |
| sad | 195 | 745 | 1786 | 2787 | 6774 (98.29%) |
| afraid | 154 | 672 | 1697 | 2696 | 6911 (98.33%) |
| nervous | 151 | 685 | 1732 | 2748 | 6692 (98.28%) |
| surprised | 153 | 666 | 1709 | 2724 | 7086 (98.22%) |

- 90% of the write needs 1472 to 2025 neurons (12% to 16% of the layer).
- 50% needs 87 to 261 neurons.
- Greedy stops by itself at about 6800 neurons (98%).
  - After that, no single neuron helps. The rest only cancel as a group.
- Happy's top neuron is 2862 (the laughter neuron of `exp/probe_kv_l25`). It carries 8.9% of the write alone.
  - Happy's top 10 neurons carry 26%. The set carries 83% along the write.
- Same sizes as `exp/residual_sets` (1499 to 2232), with a different baseline and stop. The size comes from the write, not the method.

## Limits

- The set is picked and scored on the same stories.
- A neuron can be in the set for a side direction, not for the emotion. `write_share_pct` shows which: near 0 or negative.
