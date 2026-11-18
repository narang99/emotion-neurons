# The emotion vectors across depth: who builds them, and where

Run 2026-11-19. Configs `configs/layers_all.yaml`. Outputs: `results/layers_all/{journey.json, journey.md, charts/}`;
the pooled sums (5 GB) are at `s3://narang99-private/emotion-neurons/layers_all/runs`.

```
python scripts/extract_layers.py --config configs/layers_all.yaml   # L4, 4 min
python scripts/journey.py        --config configs/layers_all.yaml   # CPU, 1 min
```

## Setting

- Same stories, split and token range as `stage2_l25` (100 train / 20 held-out per emotion, tokens 50+).
- Every layer's `resid_mid`, `resid_post` and `mlp_act`, summed per story; plus the embeddings.
- Vector at layer l: the `mean_contrast` recipe on `resid_post`, denoised with that layer's neutral PCs.
- The stream is a sum, so the vector splits exactly: `v_l = c(embed) + sum_{k <= l} (c(attention_k) + c(mlp_k))`.
  - `c(.)` is the emotion contrast: the emotion's train mean minus the 12-emotion mean.
  - Attention is measured as a lump per layer (`resid_mid_l - resid_post_{l-1}`), not decomposed.
- Shares are along the vector's own unit direction, as a fraction of its length; they sum to exactly 1.

## Log

- **Check:** layer 25 reproduces Stage 2.
  - The pooled sums match `stage2_l25` to a relative 2e-11 (resid) and 2e-8 (mlp_act).
  - Held-out accuracy at layer 25 is 0.83 (Stage 2: 0.829); the MLP adds 9.8% of |v| (Stage 2 analysis: 8-11%).
- **Round 1: where is the emotion decodable?** From the first layer, best in the middle.
  - Held-out 12-way accuracy: 0.64 at layer 0, 0.78 at 9, 0.85 at 19, peak 0.87 at 22, then down to 0.71-0.75.
  - Layer 0 already decodes 0.64: the stories' words differ by emotion.
- **Round 2: who builds the vector?** Attention early and in the middle, MLPs late.
  - Layer-25 vector: all attention layers 0.66, all MLPs 0.34, embeddings 0.00.
    - Largest single components: MLP 25 +0.098, MLP 24 +0.095, attention 25 +0.090, MLP 23 +0.089.
  - Last-layer vector: attention 0.40, MLPs 0.60.
    - Largest: MLP 35 +0.073, MLP 34 +0.071, MLP 33 +0.065, MLP 30 +0.063, MLP 31 +0.062.
  - Share added at each layer along its own vector: attention 0.15-0.29 in layers 1-12, MLP below 0.06 in 12-21.
    - MLPs pick up from layer 22: 0.10-0.14 per layer to the end.
- **Round 3: does the direction persist?** It turns slowly.
  - Cosine with the previous layer: 0.82 at layer 5, 0.92-0.95 from layer 15 on.
  - Cosine with the layer-25 vector: 0.43 at layer 10, 0.73 at 20, 0.80 at 30, 0.64 at 35.
  - Cosine with the last-layer vector: 0.29 at layer 10, 0.64 at 25.
- **Round 4: how aligned is each layer's MLP write with the vector?** Weakly in the middle.
  - Cosine of the MLP's emotion write with the layer's vector: 0.88 at layer 0, 0.14 at 12, 0.02 at 15,
    0.41 at 24, 0.47 at 30.

## Surprises

- **Middle-layer MLPs barely write along the emotion vector.** Layers 12-21 add under 6% each, with cosine 0.02-0.14.
  - Attention carries the vector through the middle; the MLPs take over from layer 22.
- **Decodability peaks at layer 22, before the vector is longest.**
  - |v| keeps growing to the last layer (18.7 at 25, 73.3 at 35), but held-out accuracy falls from 0.87 to 0.73.
  - The late growth is partly along directions that separate the emotions less well.
