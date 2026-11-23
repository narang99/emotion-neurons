# Refusal direction with earlier MLPs mean-ablated

Does the refusal direction of Qwen-1 7B Chat survive when the MLPs that build it keep only their tails?
A follow-up of `goodfire-sparsity` (issue #10), on the refusal direction instead of the emotion vectors.

- Site: the residual stream at layer 18's input, last post-instruction token (`\n` after `assistant`).
  - The `refusal_qwen1` choice: the best of 120 candidates by bypass score, with induce > 0 and KL < 0.1.
- Direction: mean of the 128 harmful train prompts minus mean of the 128 harmless ones, at the site.
- Ablation: the MLP neurons of layers k to 17 are mean-ablated, for k = 0, 4, 8, 12, 16, 17.
  - band: values strictly between the low and high pole get the neuron's mean (p3/p97 and p5/p95, the `goodfire-sparsity` poles);
  - random control: at every token, as many neurons as the band ablates there, chosen at random, get their means.
- Per condition:
  - cosine of the direction with the intact model's, and its norm;
  - the ablated model's own refusal rate: greedy replies (64 tokens) to the 32 harmful and 32 harmless val prompts, phrase match as in `refusal_qwen1`;
  - the share of cells ablated in layers k to 17, counted on the train prompts.
- Floor: the cosine between directions of two random halves of the intact train prompts, 10 splits.

## Run

On a Colab L4, 615s.

```
python scripts/refusal_ablation.py --config experiments/refusal-ablation/refusal_ablation.yaml
python experiments/refusal-ablation/plot.py   # report.html, CPU
```

Outputs in `refusal_ablation/`:

- `directions.safetensors`: per condition, `<condition>/harmful_mean`, `<condition>/harmless_mean` and `<condition>/direction`, fp32 [4096].
  - Conditions: `intact`, and `<band>/<kind>/start<k>`, e.g. `p3_p97/band/start0`. For a later steering test.
- `results.json`: the split-half floor, the intact direction's cosine with `refusal_qwen1`, one record per condition.
- `replies.json`: every val reply, per condition and category.
- `report.html`: cosine and both refusal rates against k, from `plot.py`.

## Results

- The intact direction equals the committed `refusal_qwen1` direction: cosine 0.9999965.
- Split-half floor: mean 0.981, min 0.975. Halves of the same prompts agree to about 0.98.
- Intact refusal: 93.8% of harmful prompts, 6.2% of harmless ones (as in `refusal_qwen1`).

| condition | cells ablated | cosine | norm | refusal, harmful | refusal, harmless |
|---|---|---|---|---|---|
| intact |  | 1.000 | 56.7 | 93.8% | 6.2% |
| p3/p97 band, layers 0 to 17 | 93.8% | 0.906 | 51.7 | 78.1% | 9.4% |
| p3/p97 band, layers 4 to 17 | 93.3% | 0.914 | 51.6 | 78.1% | 6.2% |
| p3/p97 band, layers 8 to 17 | 93.5% | 0.913 | 53.2 | 96.9% | 15.6% |
| p3/p97 band, layers 12 to 17 | 94.5% | 0.928 | 61.5 | 96.9% | 6.2% |
| p3/p97 band, layers 16 to 17 | 95.4% | 0.970 | 60.4 | 93.8% | 6.2% |
| p3/p97 band, layers 17 to 17 | 95.5% | 0.984 | 57.8 | 90.6% | 6.2% |
| p3/p97 random, layers 0 to 17 | 87.1% | 0.001 | 39.0 | 0.0% | 0.0% |
| p3/p97 random, layers 4 to 17 | 88.1% | 0.169 | 20.7 | 3.1% | 0.0% |
| p3/p97 random, layers 8 to 17 | 88.1% | 0.346 | 29.4 | 6.2% | 0.0% |
| p3/p97 random, layers 12 to 17 | 93.6% | 0.610 | 36.1 | 43.8% | 3.1% |
| p3/p97 random, layers 16 to 17 | 95.5% | 0.805 | 48.0 | 90.6% | 6.2% |
| p3/p97 random, layers 17 to 17 | 95.5% | 0.897 | 51.3 | 90.6% | 6.2% |
| p5/p95 band, layers 0 to 17 | 90.5% | 0.942 | 54.3 | 84.4% | 9.4% |
| p5/p95 band, layers 4 to 17 | 89.8% | 0.946 | 53.8 | 84.4% | 6.2% |
| p5/p95 band, layers 8 to 17 | 90.1% | 0.945 | 54.1 | 96.9% | 3.1% |
| p5/p95 band, layers 12 to 17 | 91.3% | 0.954 | 59.4 | 96.9% | 12.5% |
| p5/p95 band, layers 16 to 17 | 92.5% | 0.981 | 59.2 | 93.8% | 6.2% |
| p5/p95 band, layers 17 to 17 | 92.6% | 0.990 | 57.5 | 90.6% | 6.2% |
| p5/p95 random, layers 0 to 17 | 83.6% | 0.036 | 14.0 | 0.0% | 0.0% |
| p5/p95 random, layers 4 to 17 | 84.1% | 0.249 | 17.9 | 12.5% | 3.1% |
| p5/p95 random, layers 8 to 17 | 84.1% | 0.424 | 26.6 | 18.8% | 0.0% |
| p5/p95 random, layers 12 to 17 | 90.2% | 0.662 | 35.3 | 59.4% | 6.2% |
| p5/p95 random, layers 16 to 17 | 92.6% | 0.827 | 48.1 | 90.6% | 6.2% |
| p5/p95 random, layers 17 to 17 | 92.6% | 0.909 | 51.4 | 93.8% | 6.2% |

- The direction survives the band ablation of every MLP before it.
  - All of layers 0 to 17 at p3/p97: cosine 0.906, the model still refuses 78.1% of harmful prompts.
  - At p5/p95: cosine 0.942, refusal 84.4%.
  - From k = 16 on, the band cosine reaches the split-half floor (min 0.975), except p3/p97 at k = 16 (0.970).
- The random control destroys it.
  - All of layers 0 to 17: cosine 0.001 (p3/p97) and 0.036 (p5/p95), refusal 0%.
  - At k = 0 the random-ablated model breaks down rather than complies: its replies are blank lines, repeated tokens and fragments, e.g. `' the in idea\n\n the the kids...'`.
  - Its refusal recovers only as fewer layers are ablated: 43.8% (p3/p97) and 59.4% (p5/p95) at k = 12, about 91% at k = 16.
- So the tails of layers 0 to 17 carry most of the refusal direction.
  - Derived: ablating everything between the poles in layers 0 to 17 puts the cosine 0.075 (p3/p97) and 0.039 (p5/p95) below the split-half mean.
- Over-refusal stays near the intact 6.2% under the band ablation, with two blips: 15.6% (p3/p97, k = 8) and 12.5% (p5/p95, k = 12), 5 and 4 prompts of 32.
- The band ablation of layers 0 to 17 lets some harmful prompts through: refusal drops from 93.8% to 78.1% (p3/p97), 5 more of 32 prompts with no refusal phrase.
- This fits the doubt about the prompt KL in `goodfire-sparsity`: there, ablating all 32 layers at p3/p97 gave a harmful-prompt KL of 5.0, a large shift by that measure.
  - Here, ablating layers 0 to 17 at p3/p97 still leaves 78.1% refusal. The two runs ablate different layers, so this is a hint, not a test.
- Cell shares here are measured on the train prompts, so they differ slightly from the `goodfire-sparsity` text numbers.
