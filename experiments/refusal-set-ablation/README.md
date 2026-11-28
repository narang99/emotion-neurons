# Mean-ablating the top refusal neurons in the model

Do the MLP neurons that build the refusal direction carry the model's refusal? Ablate the top k in the model and measure refusal by generation.

- Model: Qwen-1 7B Chat. Direction r: layer 18 input, last post-instruction token, as in `refusal-sparse-set`.
- Step 1, `greedy_order.yaml` (`direction_sets.py`): the greedy order of last-token MLP neurons of layers 0 to 17 that rebuild r, up to 10,000 neurons, and the random control order.
- Step 2, `refusal_set_ablation.yaml` (`refusal_set_ablation.py`): the first k neurons of each order are set to their band mean (`goodfire-sweep-final` poles) at every position, prompt and reply (`ablation.SetAblation`).
  - Per condition: the direction recomputed at layer 18 and its cosine with r, and the refusal rate on the 32 harmful and 32 harmless val prompts (greedy replies, 64 tokens, phrase match).

## Run

On a Colab L4: step 1 148s, step 2 279s.

```
python scripts/direction_sets.py --config experiments/refusal-set-ablation/greedy_order.yaml
python scripts/refusal_set_ablation.py --config experiments/refusal-set-ablation/refusal_set_ablation.yaml
```

- `refusal_set_ablation/results.json`: one record per condition: cosine, norm, refusal rates.
- `greedy_order/` and `replies.json` were written on the runtime and were not kept.

## Results

| k | greedy: cosine | greedy: harmful refused | random: cosine | random: harmful refused |
|---|---|---|---|---|
| 0 | 1.000 | 93.8% | 1.000 | 93.8% |
| 10 | 0.915 | 81.3% | 1.000 | 93.8% |
| 76 | 0.781 | 56.3% | 1.000 | 93.8% |
| 300 | 0.709 | 50.0% | 1.000 | 90.6% |
| 1,000 | 0.652 | 43.8% | 0.999 | 87.5% |
| 3,550 | 0.585 | 40.6% | 0.991 | 84.4% |
| 10,000 | 0.483 | 46.9% | 0.980 | 87.5% |

- Harmless prompts: 6.3% refused in every condition, so the ablation adds no over-refusal.
- The top 76 greedy neurons cut refusal from 93.8% to 56.3%. 76 random neurons change nothing.
- The effect plateaus at 40 to 47% from about 300 neurons on.
  - Ablating r itself (`refusal_qwen1`) brings refusal to 9.4%. So these MLP neurons carry part of refusal, and something else carries the rest.
  - The summed MLP write at the site rebuilds only 39.4% of r (`refusal-sparse-set`). Attention writes most of the rest.
- The ablated model stays coherent. At k = 1000 it writes a SQL injection payload and describes flaying, but still refuses the eating-disorder prompt.
- Follow-up: issue #38, weigh the attention writers against these neurons.
