# The sparsest neuron set behind the refusal direction

How much of the refusal direction of Qwen-1 7B Chat do a few MLP neurons carry, measured by how well it still classifies prompts without them?

- Site: the residual stream at layer 18's input, last post-instruction token, as in `refusal-ablation`.
- Direction r: mean of the 128 harmful train prompts minus mean of the 128 harmless ones, at the site.
- Pool: every neuron of the MLPs of layers 0 to 17 (18 x 11008 = 198,144), at the last token only.
  - A neuron's part of r: `Δā_n w_n`, its harmful-minus-harmless mean activation times its `W_down` column.
  - Only the direct write at the last token counts, not paths through attention from earlier tokens.
  - Mean ablation sets a neuron to one value for both categories, so it removes exactly this part from r.
  - The band mean and the all-token mean give the same result here, because the mean cancels in the difference.
- Greedy: the archive greedy set (`sparse.greedy_set`), with r as the target. Each step adds the neuron that rebuilds the most of r that is still missing.
- Curve: for N neurons, remove the first N parts from r.
  - Each prompt's cosine with the reduced vector classifies it as harmful or harmless.
  - The threshold is fitted on the train prompts. Accuracy is on the 32 + 32 val prompts.
  - The prompts' activations are the intact model's. Only the vector changes.
  - Control: the same curve for a random order of the pool.

## Run

```
python scripts/direction_sets.py --config experiments/refusal-sparse-set/refusal_sparse_set.yaml   # GPU, L4
```

Outputs in `refusal_sparse_set/`:

- `results.json`: the MLP share of r, greedy set sizes, the top 100 greedy neurons, every curve.
- `orders.safetensors`: r, each order (`layer * 11008 + neuron`), and the greedy rebuilt share per step.

On a Colab L4, 722s: capture 82s, greedy to its 50,000-step cap about 640s.

## Results

- Intact r: norm 56.7. Val accuracy 96.9% (2 of 64 wrong), train 99.6%.
- The summed MLP write of layers 0 to 17 at the last token:
  - cosine 0.664 with r;
  - projection 58.4% of |r|²;
  - removing all of it rebuilds 39.4% of r. Attention and the embedding write the rest.
- Greedy set sizes: 50% of r at 76 neurons, 75% at 859, 90% at 3550, 95% at 6528.
- The top 100 greedy neurons are all in layers 12 to 17: 47 in layer 17, 24 in layer 16, 14 in layer 15.
  - First: L15 n7782 and L15 n8737. L17 n5634 and L17 n799 are #5 and #6.
  - The `refusal_qwen1` prompt view also found L17 n5634 and L17 n799, as harm-category detectors.

| removed | greedy: rebuilt | greedy: cosine | greedy: val acc | random: rebuilt | random: cosine | random: val acc |
|---|---|---|---|---|---|---|
| 0 | 0.000 | 1.000 | 96.9% | 0.000 | 1.000 | 96.9% |
| 1 | 0.074 | 0.987 | 98.4% | 0.000 | 1.000 | 96.9% |
| 10 | 0.275 | 0.932 | 98.4% | 0.000 | 1.000 | 96.9% |
| 100 | 0.527 | 0.801 | 100% | 0.000 | 1.000 | 96.9% |
| 1,000 | 0.767 | 0.682 | 100% | -0.003 | 0.999 | 96.9% |
| 10,000 | 0.975 | 0.583 | 100% | 0.008 | 0.985 | 96.9% |
| 30,000 | 1.000 | 0.474 | 100% | 0.038 | 0.960 | 96.9% |
| 50,000 | 1.000 | 0.381 | 100% |  |  |  |
| 100,000 |  |  |  | 0.220 | 0.811 | 96.9% |
| 198,144 (all) |  |  |  | 0.394 | 0.534 | 100% |

- Val accuracy never drops. It stays between 96.9% and 100% for every N in both orders.
- Greedy rebuilds all of r, but the whole MLP write rebuilds only 39.4%.
  - 198,144 parts in 4096 dimensions span the space, so greedy can fit any vector.
  - Past about 40%, the greedy set fits r with MLP parts that do not write it in the model.
- At 30,000 greedy neurons the reduced vector keeps under 0.1% of |r|². It still classifies all 64 val prompts correctly.
- So this measure cannot tell which neurons the direction needs.
  - The prompts' activations still hold every MLP write. Harmful and harmless prompts separate along many directions, so almost any leftover vector still separates them.
- `orders.safetensors` (2.4 MB) was written on the runtime and is not in this folder yet.
