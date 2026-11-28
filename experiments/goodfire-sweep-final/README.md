# Goodfire sparsity on Qwen-1 7B Chat: sweep with the in-band mean

The band ablation of `goodfire-sparsity`, with one change: in-band values get the in-band mean (issue #31).

- The model's normal input: no BOS added, and no positions skipped.
  - The poles, the ablation and the scoring all use every position (`skip_first: 0`, ablation positions `all`).
- Sweep: the band ablation of layers k to 31, for k = 0, 4, 8, ..., 28 and 31. Layers before k stay intact.
- Random control at k = 0: as many neurons per token as the band, chosen at random.
- No refusal-prompt KL.

## Runs

All on a Colab L4.

```
python scripts/poles.py --config experiments/goodfire-sweep-final/poles_qwen1.yaml          # p3, p97, band mean; 1687s
python scripts/poles.py --config experiments/goodfire-sweep-final/poles_qwen1_p5.yaml       # p5, p95, band mean; 1585s
python scripts/ablation.py --config experiments/goodfire-sweep-final/ablation_sweep.yaml    # 533s
python scripts/ablation.py --config experiments/goodfire-sweep-final/ablation_sweep_p5.yaml # 532s
python experiments/goodfire-sweep-final/plot.py                                             # report.html, CPU
```

- `poles_qwen1/`, `poles_qwen1_p5/`: thresholds over 210,676 tokens, every position included.
- `ablation_sweep/`, `ablation_sweep_p5/`: `ablation.json`, one record per ablation.
- `report.html`: loss change and KL against the first ablated layer, both bands.

## Results

Intact loss 2.142 on 104,926 held-out positions.

| first ablated layer k | p3/p97 loss change | p3/p97 KL | p5/p95 loss change | p5/p95 KL |
|---|---|---|---|---|
| 0 | +1.231 | 1.446 | +0.539 | 0.662 |
| 4 | +1.027 | 1.224 | +0.440 | 0.561 |
| 8 | +0.889 | 1.067 | +0.390 | 0.503 |
| 12 | +0.725 | 0.865 | +0.323 | 0.413 |
| 16 | +0.490 | 0.560 | +0.253 | 0.296 |
| 20 | +0.333 | 0.349 | +0.191 | 0.198 |
| 24 | +0.258 | 0.244 | +0.154 | 0.143 |
| 28 | +0.224 | 0.196 | +0.139 | 0.117 |
| 31 | +0.149 | 0.125 | +0.097 | 0.079 |
| random, k = 0 | +7.794 | 8.475 | +7.803 | 8.487 |

- Cells ablated: 94% (p3/p97) and 90% (p5/p95) at every k.
- The loss change falls steadily as fewer early layers are ablated.
- Layer 31 alone costs +0.097 at p5/p95, about Goodfire's whole-model +0.091.
- Intact loss is higher than in `goodfire-sparsity` (2.062), because positions 0 to 3 are now scored.
