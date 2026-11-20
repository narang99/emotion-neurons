# Signal set and cancelling rest

Date: 2026-11-20. Branch: `stack/06-raw-sets`. Tag: `exp/residual_sets`. Status: kept.

Config: `configs/residual_sets.yaml`. Results: `results/residual_sets/residual_sets.json`.

```
python scripts/residual_sets.py --config configs/residual_sets.yaml --device cpu   # CPU, 30 s, from the signal_noise cache
```

## Question

- Can the neurons be split into a signal set S that rebuilds the emotion write, and a rest that cancels to zero?
- "Rebuilds" means direction and length: the leftover `|d - sum_S u_n| / |d|` is small. The leftover is exactly the rest's sum.

## Method

- Target: `d = W_down (abar_own - abar_other)`, raw story-mean activations, train stories. Other = 11 emotions and neutral.
- Greedy (`decomposition.residual`): add the neuron that shrinks the leftover most.
- Stop: where the leftover reaches the train gap's own sampling noise (0.24 to 0.37 of its length).
  - Past that point the set fits story noise.
- Held-out: the rest's held-out sum against the noise floor (what it would be if the rest summed to exactly zero).
- Controls: random sets of the same size; the match sets.

## Results

| | signal set at the noise level | random set, same size | match set |
|---|---|---|---|
| neurons | 1499 to 2232 (12% to 18%) | same | 268 to 1401 |
| leftover on train | 0.24 to 0.37 (= noise) | 0.88 to 0.93 | 0.50 to 0.76 |
| held-out leftover / noise floor | 1.06 to 1.41 (mean 1.19) | | |
| held-out cosine with the gap | 0.72 to 0.90 | | |
| write from pole tokens | 0.65 to 0.76 | | |

- Leftover by set size (happy): 100 neurons 0.69, 300 0.58, 1000 0.40, 3000 0.19.
- **The split exists, but the signal set is not small.** About 1900 neurons (15%) rebuild the write to its noise level.
- **The rest (about 10,400 neurons) mostly cancel.**
  - On held-out stories their sum is 1.19 times what pure noise would give. A little signal is left in them.
- **The match sets are not signal sets.** They point the right way but leave 50% to 76% of the write in the rest.
- The signal set's write comes 65% to 76% from pole tokens.

## Limits

- Only 20 held-out stories per emotion. The held-out gap is about half noise (0.48 to 0.68 of its length).
- Greedy is not guaranteed optimal.
