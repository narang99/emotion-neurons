# Dense circuits for the layer-25 emotion score

Run 2026-11-19. Config `configs/circuits_l25.yaml`. Outputs: `results/circuits_l25/{circuits.jsonl, circuits.md}`.

```
python scripts/layer_vectors.py --config configs/layers_all.yaml    # CPU, once
python scripts/circuits.py      --config configs/circuits_l25.yaml  # L4, 3 min for 60 stories
```

## Setting

- Target: the emotion score `s_e(t) = u_e . resid_post_25(t)` at every token from 50 on, where `u_e` is the
  denoised layer-25 vector of the story's own emotion.
- 60 held-out stories, 5 per emotion.
- Every neuron of layers 0-25 gets an effect from one backward pass.
  - Attention outputs and RMSNorm scales are frozen, so paths stay MLP-only and at the same token.
  - The direct shares (neuron write `. u`, plus attention and embeddings) sum to the score exactly.
- A circuit is the top-k neurons by |effect|.
  - Faithfulness: `(s(C) - s(none)) / (s(all) - s(none))`, with every neuron outside C held at its 12-emotion mean.
  - Control: random sets of the same size.
- Emotion sets: each layer's `k_match` sets from `sparse_layers`.

## Log

- **Round 1 (`exp/circuits_l25-round1`, failed): effect against zero activation.** The circuits are not faithful.
  - Direct shares rebuild the score to 0.8% (exact up to bf16).
  - Faithfulness: -0.10 at 100 neurons, -0.12 at 1000, 0.03 at 3000, 0.24 at 10000, 0.53 at 30000.
    - Random sets: 0.01, -0.02, 0.01, 0.08, 0.04.
  - Why: `a * ds/da` counts a neuron's whole activation, including the part that is the same for every emotion.
    - 29% of the top-1000 |effect| sat in layer 0, on large, nearly constant activations.
    - The summed neuron effect (median 8882) is 6x the gap that ablation can open, `s(all) - s(none)` (1539).
    - Mean ablation leaves those neurons almost where they were, so keeping them buys nothing.
  - The membership split of these circuits (18% own emotion set, 36% other sets, 46% none) is not meaningful yet.
- **Round 2 (pending, needs an L4): effect against the mean**, `(a - mean) * ds/da`.
  - This is the first-order prediction of exactly what the check does, so ranking and check agree.
  - Code and a test are in; the run is not done.
