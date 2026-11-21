# Early-layer neurons of the emotion write: what they fire on

Date: 2026-11-19. Branch: `stack/05-causal-sets`. Tag: `exp/harvest_layers`. Status: kept.

Config: `configs/harvest_layers.yaml`. Results: `results/harvest_layers/L{0,2,8,12}.json`.
Activations: `runs/harvest_layers/L<k>/acts.npy` (S3).

```
python scripts/harvest_layers.py --config configs/harvest_layers.yaml   # L4, 5.5 minutes for 4 layers
```

## Question

- In early layers, are the neurons that write the emotion vector emotion-word detectors?

## Method

- Layers 0, 2, 8 and 12. Per layer:
  - the first 8 neurons of each emotion's sparse set (`results/sparse_layers`), 45 to 49 neurons in all;
  - named neurons (L8 n4826, n7265);
  - 40 random controls.
- One pass over the same 2M-token corpus as the layer-25 harvest (fineweb-edu, TinyStories, UltraChat).
- Per neuron:
  - the percentile thresholds;
  - the top 20 snippets;
  - the counts of the tokens at its top 100 activations;
  - `emotion_share`: the share of its top 50 tokens that are emotion words.
- The run is cached per layer. A rerun skips a layer whose JSON exists.

## Results

- **Emotion-word share of the top 50 tokens:**

  | layer | sparse-set neurons | controls |
  |---|---|---|
  | 0 | 0.006 | 0.006 |
  | 2 | 0.008 | 0.000 |
  | 8 | 0.029 | 0.002 |
  | 12 | 0.045 | 0.003 |

  - The early neurons of the emotion write are almost never emotion-word detectors. The share grows with depth.
- **What they fire on: function words, persons, and body and situation words.** Examples:
  - pronouns: L0 n317 (loving) " Her, her"; L0 n606 (loving) " he, she, we, they"; L2 n557 (angry) " you, I, your";
  - body: L0 n956 (afraid) " heart"; L8 n1381 (angry) " hot, heat";
  - posture: L2 n214 (loving) " stood, stand, standing";
  - adverbs: L0 n943 (nervous) " too"; L8 n1432 (calm) " fast, second, -Speed".
- **The two named L8 neurons are lexical:**
  - n7265: " embarrassed" (44 of the top 100), " ashamed" (35), " shy" (7);
  - n4826: " himself" (49), " herself" (28), " satisfaction", " accomplishment". It reads the frame "proud of himself".
  - Their emotion share is 0.41.

## Conclusion

- In early layers, the emotion vector is built from many context tokens (persons, body, posture), not from emotion words.
- This agrees with the word tests: only 14 of 240 held-out stories name the emotion, yet layer 0 classifies them at 0.64.
