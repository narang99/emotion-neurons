# Causal test of the layer-25 sparse sets

Date: 2026-11-19. Branch: `stack/05-causal-sets`. Tag: `exp/causal_sets`. Status: kept.

Config: `configs/causal_sets.yaml`. Results: `results/causal_sets/causal_sets.json`. Streams: `runs/causal_sets/` (S3).

```
python scripts/causal_sets.py --config configs/causal_sets.yaml   # L4, 9 minutes
```

## Question

- Do the sparse sets cause the emotion, or do they only rebuild the direction of the write?

## Method

- Sets: each emotion's `match` set (`results/sparse_l25_denoised/sets.json`).
  - It is the smallest set that matches the full MLP on held-out stories.
  - Sizes: 268 (proud) to 1401 (inspired). The mean is 728, which is 5.9% of 12288 neurons.
  - The union of all 12 sets is 4709 neurons (38%).
- Shift `dA_e`: each neuron's mean activation on emotion e's train stories, minus its 12-emotion mean.
- Edits at layer 25, at every token of each of the 240 held-out stories:
  - **remove**: `a[S] -= dA_e[S]`, on the story's own emotion;
  - **mean**: `a[S] = 12-emotion mean`;
  - **swap e -> f**: `a[S] += scale * (dA_f - dA_e)[S]`, with S the union of the two sets. Scales 1 and 4.
- S is the sparse set, 3 random sets of the same size, or all neurons.
- Readouts:
  - emotion scores at layers 25, 28, 31 and 35;
  - the model's answer to "In one word, how does the main character feel?", scored over the 12 emotion words.

## Results

- **The set carries most of what the whole MLP-25 can do. Random sets carry little.**
  - Share of the write along `delta_e` that the set's part of the shift carries: 0.40 to 0.69. Random sets: 0.02 to 0.11.
  - Swap, change of the score gap (target minus source) at layer 35, scale 1:

    | swap | set | random sets | all neurons | clean gap |
    |---|---|---|---|---|
    | happy -> sad | 5.1 | 0.4 | 7.2 | 66.8 |
    | sad -> happy | 4.9 | 0.5 | 7.0 | 54.5 |
    | calm -> afraid | 6.2 | 1.4 | 9.1 | 62.6 |
    | afraid -> calm | 6.1 | 1.1 | 8.7 | 71.2 |
    | proud -> guilty | 4.1 | 0.3 | 6.2 | 65.5 |
    | guilty -> proud | 4.1 | 0.3 | 6.2 | 57.5 |

  - The set gives about 70% of the change of all neurons, with 6% of the neurons.
  - Random sets of the same size give 5% to 20% of the set's change.
- **But MLP-25 alone is a small part of the emotion.**
  - Scale 1: the gap moves by about 15% of the clean gap at layer 25 (set), and by about 8% at layer 35.
  - Scale 4: the gap moves by about 58% at layer 25 and by about 30% at layer 35 (set). All neurons: about 82% and 43%.
  - No label flips to the target emotion at scale 1. At scale 4, at most 10% of stories flip (all neurons).
  - The answer gap moves by 0.2 to 0.3 nats at scale 1, and by 0.8 to 2.1 nats at scale 4. The typical margin of the true answer is 6.7 nats. No answer flips.
- **Removal moves accuracy a little.**
  - Accuracy at layer 25: clean 0.82; set removed 0.80; random sets removed 0.82; all neurons removed 0.78.
  - The true emotion's score drops by 7% (set) and 10% (all neurons) at layer 25. Random sets: 1%.
- The continuations did not show the emotion. The model answered with a quiz format after "That evening,". They are in the JSON, but they are not useful.

## Conclusion

- The sparse sets are causal. They carry about 70% of MLP-25's causal effect on later emotion scores, and much more than random sets of the same size.
- But MLP-25 is not where the emotion is decided. Its whole emotion write moves later scores by only about 10%.
  - This agrees with the journey result: at layer 25 the vector is 66% attention and 34% MLP, and later layers add more.

## Mistakes

- The first run stopped at 80 of 240 stories because the VM was released. The second run completed.
