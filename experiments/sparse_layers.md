# Sparse sets of every layer's MLP emotion write

Run 2026-11-19. Config `configs/sparse_layers.yaml`; needs `runs/layers_all` from `journey_all`.
Outputs: `results/sparse_layers/{sparse_layers.json, sparse_layers.md, charts/k_match.svg}`.

```
python scripts/sparse_layers.py --config configs/sparse_layers.yaml   # CPU, 13 min (W_down of 36 layers cached)
```

## Setting

- The method of `top64_l25.md` at every layer: plain greedy on the MLP's emotion write, train stories.
- `k_match`: the smallest prefix whose held-out transfer reaches the full MLP's (the ceiling).
- `k for cos 0.9`: the smallest prefix whose train cosine with the write reaches 0.9.
- W_down per layer is read from the Hub and cached as bf16; the write rebuilds from `W_down dA` to 0.06-0.20%.

## Log

- **Check: layer 25 against `sparse_l25_raw`.** Same for 7 of 12 emotions; within 4% for the rest.
  - The inputs differ by about 1e-8 (the all-layer sums against Stage 2's), and the greedy orders split on
    near-ties after 78 to 1134 picks.
  - Near `k_match` the transfer curve stays within 0.001 of the ceiling over dozens of neurons.
  - So `k_match` is defined to about +/-4%: inspired 1212 vs 1229, desperate 874 vs 899, angry 1279 vs 1233.
- **Round 1: how many neurons does each layer's write need?** Most in the middle, fewest at the ends.
  - `k for cos 0.9` (median over emotions): 445 at layer 0, 734 at 5, 1178 at 10, 1216 at 12, 1102 at 20,
    928 at 25, 565 at 30, 575 at 33, 38 at 35.
  - The last layer's write is carried by a handful of neurons: 38 for cos 0.9.
- **Round 2: when does a sparse set generalise as well as the full MLP?** Not at every layer.
  - All 12 emotions match the ceiling at layers 2-3, 5-7, 10, 20, 22-23, 25-27, 29-33.
  - In layers 12-16 only 4-6 of 12 do within 2500 neurons, and at layer 35 only 3.
  - Held-out ceiling (median): 0.58-0.68 in layers 0-8, 0.73-0.80 from layer 12 on, 0.86 at layer 35.
  - Median `k_match`: 119 at layer 0, 386 at 5, 759 at 10, 1194 at 20, 761 at 25, 424 at 30.

## Surprises

- **The middle layers are the densest and the least reliable.**
  - Layers 12-19 need the most neurons for cos 0.9 (about 1000-1200), and their sparse sets often never reach the
    full MLP's held-out score.
  - That matches `journey_all`: those MLPs barely write along the emotion vector.
- **The last layer's emotion write is sparse:** 38 neurons reach cos 0.9.
