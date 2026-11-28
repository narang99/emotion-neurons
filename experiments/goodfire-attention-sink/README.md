# Goodfire sparsity on Qwen-1 7B Chat: band mean and attention sink

Two fixes to the band ablation of `goodfire-sparsity`, checked one after the other (issues #31 and #35).

- Band mean (`experiments/goodfire-band-mean/`, issue #31):
  - Before: in-band values got the neuron's mean over all tokens, tails included. The kept tails counted twice.
  - After: in-band values get the mean over the in-band values only (`poles.mean: {kind: band}`).
- Attention sink (this folder, issue #35):
  - Before: the hooks ablated every position, the first 4 included. The poles and the scoring leave those out.
  - After: the first 4 positions of each row stay intact (`positions: {kind: skip_first, count: 4}`), on the band-mean thresholds.

## Runs

All on a Colab L4.

```
python scripts/poles.py --config experiments/goodfire-band-mean/poles_qwen1.yaml         # p3, p97, band mean; 1655s
python scripts/poles.py --config experiments/goodfire-band-mean/poles_qwen1_p5.yaml      # p5, p95, band mean; 1587s
python scripts/ablation.py --config experiments/goodfire-band-mean/ablation_full.yaml    # 192s
python scripts/ablation.py --config experiments/goodfire-band-mean/ablation_full_p5.yaml # 189s
python scripts/ablation.py --config experiments/goodfire-attention-sink/ablation_full.yaml    # 259s
python scripts/ablation.py --config experiments/goodfire-attention-sink/ablation_full_p5.yaml # 194s
```

## Results

All 32 layers mean-ablated. Intact loss 2.062 on 102,998 held-out positions.

| band | run | mean | positions | cells ablated | loss change | KL per position | random loss change | random KL |
|---|---|---|---|---|---|---|---|---|
| p3/p97 | `goodfire-sparsity` | all tokens | all | 94.9% | +1.644 | 1.907 | +8.323 | 8.945 |
| p3/p97 | `goodfire-band-mean` | band | all | 94.0% | +1.245 | 1.454 | +7.882 | 8.498 |
| p3/p97 | `goodfire-attention-sink` | band | skip first 4 | 92.2% | +1.232 | 1.418 | +7.203 | 7.814 |
| p5/p95 | `goodfire-sparsity` | all tokens | all | 91.3% | +0.812 | 0.978 | +8.358 | 8.982 |
| p5/p95 | `goodfire-band-mean` | band | all | 89.9% | +0.547 | 0.665 | +7.892 | 8.511 |
| p5/p95 | `goodfire-attention-sink` | band | skip first 4 | 88.2% | +0.544 | 0.641 | +7.088 | 7.690 |
| P5/P95 | Goodfire, Qwen3.6-35B-A3B | | | 88.8% | +0.091 | 0.162 | +2.328 | 2.511 |

- The band mean cuts the loss change by 24% (p3/p97) and 33% (p5/p95).
- Keeping the attention-sink positions intact cuts it by 1% or less more.
  - The ablated share drops about 1.7 points, close to the 4 of 256 positions now spared.
- With both fixes, p5/p95 is still 6x Goodfire's loss change and 4x their KL.
- The band still beats the random control: 5.8x (p3/p97) and 13x (p5/p95).
- The difference left is the neurons tested.
  - Goodfire ablates routed MoE neurons only, and leaves the always-on shared experts intact.
  - Here every neuron of a dense MLP is ablated, so no MLP path stays intact.
