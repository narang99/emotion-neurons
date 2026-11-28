# The refusal direction at one MLP output, and its sparsest neuron set

`refusal-sparse-set` read the direction in the residual stream, where attention writes most of it. This experiment takes attention out of the picture.

- Site: the down_proj output of layer 17's MLP, last post-instruction token.
- Direction r: mean of the 128 harmful train prompts minus mean of the 128 harmless ones, at the site.
- Pool: layer 17's 11,008 neurons. A neuron's part of r is `Δā_n w_n`, and the parts sum exactly to r.
- Orders: greedy (`sparse.greedy_set`) and random, as in `refusal-sparse-set`.
- For the first k neurons of an order, three evaluations:
  - probe: the reduced vector `r - sum u_n` scores the intact prompts by cosine;
  - ablation: the k neurons are also mean-ablated in every prompt (band mean, `goodfire-sweep-final` poles), then scored by the reduced vector;
  - only: each prompt keeps only the k neurons' write `sum a_n w_n`, scored by projection on their parts.
- Random directions: 100 random unit vectors score the intact prompts by cosine.
- Every score gets a threshold fitted on the 256 train prompts. Accuracy is on the 32 + 32 val prompts.

## Run

On a Colab L4, 90s.

```
python scripts/direction_sets.py --config experiments/refusal-mlp-out/refusal_mlp_out.yaml   # GPU, L4
```

- `refusal_mlp_out/results.json`: greedy set sizes, top 100 neurons, the random-direction accuracies, every curve.
- `orders.safetensors` was written on the runtime and was not kept.

## Results

- r: norm 27.8. The parts sum to r (rebuilt 1.000). Intact val accuracy 98.4% (1 of 64 wrong).
- Greedy: 50% of r at 94 neurons, 75% at 428, 90% at 1180, 95% at 1984. It stops at 6216 neurons, 98.5% rebuilt.
- Top greedy neurons: n1958, n799, n5634, n3366, n9572, n2882.
  - The first four are #1, #2, #3 and #5 of the `refusal_qwen1` prompt view, its harm-category detectors.

### Random directions

- Val accuracy of 100 random directions: mean 57.0%, median 52.3%, 90th percentile 75.2%, max 89.1%.
- So a direction that knows nothing does not separate the prompts. r at 98.4% is far above that.
- The earlier probe point with all parts removed (73.4%) is a vector of float rounding error. It is a random direction, inside this range.

### Removing the top k neurons: probe and ablation

| k | greedy: cosine | greedy probe | greedy ablation | random: cosine | random probe | random ablation |
|---|---|---|---|---|---|---|
| 0 | 1.000 | 98.4% | 98.4% | 1.000 | 98.4% | 98.4% |
| 10 | 0.886 | 96.9% | 96.9% | 1.000 | 98.4% | 98.4% |
| 100 | 0.736 | 100% | 100% | 0.982 | 96.9% | 96.9% |
| 1,000 | 0.486 | 100% | 100% | 0.948 | 98.4% | 98.4% |
| 3,000 | 0.325 | 100% | 100% | 0.859 | 100% | 100% |
| 6,216 | 0.263 | 100% | 100% |  |  |  |
| 10,000 |  |  |  | 0.353 | 100% | 100% |
| 11,008 (all) |  |  |  | 0.000 | 73.4% | 50.0% |

- Ablation in the prompts gives the same accuracy as the probe at every k, within one prompt.
- After the 6216 greedy neurons are ablated, the remaining 4792 neurons still classify all 64 val prompts.
- Only ablating every neuron brings accuracy to 50%: all prompts are then the same vector.

### Keeping only the first k neurons

| k | greedy: val acc | random: val acc |
|---|---|---|
| 1 | 90.6% | 54.7% |
| 3 | 96.9% | 59.4% |
| 10 | 98.4% | 78.1% |
| 30 | 98.4% | 93.8% |
| 100 | 96.9% | 92.2% |
| 300 | 96.9% | 98.4% |
| 1,000 to 11,008 | 96.9% | 96.9% |

- n1958 alone classifies 90.6% of val prompts. The top 3 greedy neurons reach 96.9%, the top 10 reach 98.4%.
- Random neurons need more: 10 give 78.1%, 30 give 93.8%, 300 give 98.4%.

### In-band neurons only

- Config `band_subsets`: 20 random sets of 30 neurons, poles p3/p97 of `goodfire-sweep-final`.
  - Set 0 is the first 30 neurons of the `random_seed0` order. Sets 1 to 19 are drawn with seed + 1.
  - A neuron is in band when its last-token value stays strictly between its poles on all 256 train prompts.
  - The `only` evaluation scores the in-band part, the tail part, and 100 random drops of the same size as the tail part.
- Few neurons stay in band at this token. A set of 30 has 1 to 6 in-band neurons, 3.65 on average.

| over 20 sets | neurons, mean | mean val accuracy |
|---|---|---|
| all 30 neurons | 30 | 86.5% |
| tail neurons only | 26.35 | 86.6% |
| in-band neurons only | 3.65 | 67.6% |
| random neurons, as many as the in-band part | 3.65 | 71.7% |

- Compare rows of the same size. The tail part against the in-band part is 26 neurons against 3.65.
- In-band neurons do slightly worse than random neurons of the same count: 67.6% against 71.7%, below the random mean in 13 of 20 sets.

### Contribution and rank of the random neurons

- Per neuron, `results.json` has its tail rate, its part projected on the unit direction, its rank by projection among 11,008, and its greedy rank.
- Over the 600 neurons of the 20 sets:
  - Spearman correlation of tail rate with |projection|: 0.49.
  - The 20 neurons ranked 300 or better by projection reach a tail on 44% of prompts (median). The others: 2.3%.
  - Mean |projection|: 0.0041 for neurons that reach a tail, 0.0007 for in-band neurons.
- 15 of 20 sets contain a neuron ranked 300 or better. Their all-30 accuracy is 87.9%, against 82.2% for the other 5 sets.
- Set 0, the original set, scores 93.8% mostly through two such neurons: n9257 (projection rank 12, tail rate 52%) and n8490 (rank 35, tail rate 45%).
- So the large contributors to the direction are the neurons that often sit at their poles.
  - A random set of 30 still reaches about 82% from many small contributors, without any of them.

### What it means

- The harm signal at layer 17's output is concentrated and redundant at once.
  - Concentrated: a handful of top neurons carry almost all of it.
  - Redundant: any 30 random neurons out of 11,008 also carry most of it, and the neurons left after removing the greedy set still carry all of it.
- So removing neurons, from the vector or from the prompts, cannot bring a linear readout below the intact accuracy at this site.
- Whether the model's refusal depends on the top neurons is a question for an ablation in the model with a behavioural measure. This experiment does not answer it.
