# Sparse neuron sets for the MLP emotion write of every layer

Date: 2026-11-19. Branch: `stack/02-sparse-layers` (PR #3). Tag: `exp/sparse_layers`.

Config: `configs/sparse_layers.yaml`. It needs `runs/layers_all` from `journey_all`.
Results: `results/sparse_layers/sparse_layers.json`, `sparse_layers.md`, `charts/k_match.svg`.

```
python scripts/sparse_layers.py --config configs/sparse_layers.yaml   # CPU, 13 min
```

## Question

- How many neurons does each layer's MLP need to write its emotion vector?
- Does this number change with depth?

## Method

- We used the method of `top64_l25.md` at all 36 layers.
  - Greedy selection: each step adds the neuron that increases the cosine with the MLP write most.
  - The script computes the greedy steps exactly from a Gram matrix.
- We measured two set sizes for each layer and emotion:
  - `k for cos 0.9`: the smallest set whose train cosine with the write is 0.9 or more.
  - `k_match`: the smallest set whose held-out transfer reaches the transfer of the full MLP (the ceiling).
- The script reads W_down for each layer from the Hub. It keeps a bf16 copy (100 MB each).
- Check: the write rebuilds from `W_down dA` with an error of 0.06-0.20% at all layers.

## What we did, step by step

1. We added `shifts_from_means`. It gives the activation shifts from the means of any layer.
2. We ran layer 25 only and compared it with `sparse_l25_raw` (the earlier layer-25 result).
3. We downloaded W_down for all 36 layers in the background (3.6 GB).
4. We ran all 36 layers (13 minutes).
5. We found a bug in the summary table and ran the script again (see "Mistakes").

## Results

- **Check: layer 25 against `sparse_l25_raw`.** 7 of 12 emotions are the same. The other 5 are within 4%.
  - The inputs differ by about 1e-8. They come from the all-layer sums, not the Stage 2 sums.
  - The greedy orders are the same for the first 78 to 1134 steps. Then near-ties go a different way.
  - Near `k_match`, the transfer curve stays within 0.001 of the ceiling for many neurons.
  - Thus, `k_match` has an uncertainty of about +/-4%.
    - Examples: inspired 1212 against 1229; desperate 874 against 899; angry 1279 against 1233.
- **Result 1: the middle layers need the most neurons. The ends need the fewest.**
  - `k for cos 0.9` (median over the 12 emotions):
    - 445 at layer 0;
    - 734 at layer 5;
    - 1178 at layer 10;
    - 1216 at layer 12;
    - 1102 at layer 20;
    - 928 at layer 25;
    - 565 at layer 30;
    - 38 at layer 35.
  - The last layer writes its emotion part with few neurons. 38 neurons give cosine 0.9.
- **Result 2: a sparse set does not always generalise as well as the full MLP.**
  - All 12 emotions reach the ceiling at 17 of 36 layers: 2-3, 5-7, 10, 20, 22-23, 25-27, 29-33.
  - At layers 12-16, only 4-6 emotions reach it within 2500 neurons.
  - At layer 35, only 3 emotions reach it.
  - The held-out ceiling (median): 0.58-0.68 at layers 0-8; 0.73-0.80 from layer 12; 0.86 at layer 35.

## What worked

- The greedy method from layer 25 worked at every layer without change.
- Downloading W_down once in the background made the full run fast (13 minutes on a laptop).

## What did not work, and mistakes

- **Mistake: the summary table had a wrong average.**
  - The column "cos at match" averaged over all 12 emotions.
  - It counted an emotion with no match as 0.
  - Fix: average over the emotions that match only. We ran the script again.
- **Limit: `k_match` is not reliable in layers 12-19.**
  - Many emotions never reach the ceiling there.
  - Use `k for cos 0.9` to compare the layers.
- **Limit: `k_match` is noisy by about +/-4%,** as the layer-25 check shows.

## Surprises

- The middle layers (12-19) are the most dense, and their sparse sets generalise the least.
  - This agrees with `journey_all`: these MLPs almost do not write along the emotion vector.
- The last layer is very sparse: 38 neurons give cosine 0.9.
