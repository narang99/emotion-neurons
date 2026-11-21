# The emotion vectors across all 36 layers

Date: 2026-11-19. Branch: `stack/01-journey` (PR #2). Tag: `exp/journey_all`.

Config: `configs/layers_all.yaml`.
Results: `results/layers_all/journey.json`, `journey.md`, `charts/*.svg`.
Large data (5 GB): `s3://narang99-private/emotion-neurons/layers_all/runs`.

```
python scripts/extract_layers.py --config configs/layers_all.yaml   # L4, 4 min
python scripts/journey.py        --config configs/layers_all.yaml   # CPU, 1 min
```

## Question

- Which parts of the model make each emotion vector?
- At which layers do they make it?

## Method

- We used the same stories, split and tokens as `stage2_l25`.
  - 100 train stories and 20 held-out stories for each emotion.
  - Tokens 50 and after.
- One GPU pass recorded three sums for each story at each layer:
  - `resid_mid`: the stream after attention.
  - `resid_post`: the stream after the MLP.
  - `mlp_act`: the MLP neurons.
- The pass also recorded the embeddings (the input to layer 0).
- The hooks added up the tokens during the pass.
  - Full activations for all layers do not fit in GPU memory (about 8 GB for each batch).
- The vector at layer l is the `mean_contrast` recipe on `resid_post`.
  - We removed the neutral directions of that layer (the paper method).
- The stream is a sum of writes. Thus the vector splits exactly into parts:
  - the embeddings;
  - the attention write of each layer (`resid_mid_l - resid_post_{l-1}`);
  - the MLP write of each layer (`resid_post_l - resid_mid_l`).
- We did not split attention into heads. Each layer's attention is one part.
- For each part, we measured its length along the vector, as a fraction of the vector length.
  - These fractions add up to exactly 1. The script checks this.

## What we did, step by step

1. We wrote `extract_layers.py` and ran it on an L4 before the refactor.
   - Run time: 4 minutes.
2. We compared layer 25 with the old Stage 2 sums.
   - The difference was 2e-11 (stream) and 2e-8 (neurons). Layer 25 is the same.
3. After the refactor, we wrote the script again with the new library.
   - We ran it again on the L4.
   - The new sums are identical to the old sums (difference 0.0).
4. We wrote `journey.py` and ran it on a laptop CPU in 1 minute.

## Results

- **Check:** layer 25 agrees with Stage 2.
  - Held-out accuracy at layer 25 is 0.83. Stage 2 gave 0.829.
  - The MLP of layer 25 adds 9.8% of the vector length. The earlier analysis gave 8-11%.
- **Result 1: the emotion is readable at all layers. It is best in the middle.**
  - Held-out 12-way accuracy:
    - 0.64 at layer 0;
    - 0.78 at layer 9;
    - 0.85 at layer 19;
    - 0.87 at layer 22 (the highest);
    - 0.71-0.75 in the last layers.
  - Layer 0 gives 0.64 because the stories use different words for each emotion.
- **Result 2: attention makes most of the vector early. The MLPs make most of it late.**
  - Layer-25 vector: attention 0.66, MLPs 0.34, embeddings 0.00.
    - Largest single parts: MLP 25 (+0.098), MLP 24 (+0.095), attention 25 (+0.090), MLP 23 (+0.089).
  - Last-layer vector: attention 0.40, MLPs 0.60.
    - Largest single parts: MLP 35 (+0.073), MLP 34 (+0.071), MLP 33 (+0.065).
  - Attention adds 15-29% of the vector at each of layers 1-12.
  - The MLPs of layers 12-21 add less than 6% each.
  - From layer 22, each MLP adds 10-14%.
- **Result 3: the direction changes slowly.**
  - Cosine with the previous layer: 0.82 at layer 5. It is 0.92-0.95 from layer 15.
  - Cosine with the layer-25 vector: 0.43 at layer 10, 0.73 at layer 20, 0.80 at layer 30, 0.64 at layer 35.
- **Result 4: the middle MLPs do not write along the vector.**
  - Cosine of each MLP write with its layer's vector:
    - 0.88 at layer 0;
    - 0.14 at layer 12;
    - 0.02 at layer 15;
    - 0.41 at layer 24;
    - 0.47 at layer 30.

## What worked

- Sums taken inside the hooks. They kept the GPU memory low, and one pass gave all 36 layers.
- The exact split. The fractions add up to 1 for every emotion, so no part is missing.
- The layer-25 check against Stage 2. It found no difference.

## What did not work, and mistakes

- No failure in this experiment.
- One limit: attention is one part for each layer. We do not know which heads do the work.

## Surprises

- The middle MLPs (layers 12-21) almost do not write along the emotion vector.
  - Attention carries the vector through the middle layers.
- Accuracy is highest at layer 22, but the vector length grows until the last layer.
  - The vector length is 18.7 at layer 25 and 73.3 at layer 35.
  - Thus, the late growth is partly in directions that do not separate the emotions well.
