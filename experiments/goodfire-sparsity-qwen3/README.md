# Goodfire sparsity on Qwen3-8B

The `goodfire-sparsity` test (issue #10) repeated on Qwen3-8B, the emotion-vector model: 36 layers of 12288 neurons.

- Claim (Goodfire, "Neurons as an interpretable basis"): mean-ablating every MLP neuron value between its low and high percentile, in all layers at once, barely hurts the model.
  - Their numbers (Qwen3.6-35B-A3B, routed neurons, Table 1): P5 to P95 ablates 88.8% of cells, loss +0.091, KL 0.162.
- Same method, code and text as `goodfire-sparsity` (Qwen-1 7B Chat); see its README.
  - Poles: per neuron, p3, p5, p95, p97 and the mean of `silu(gate) * up` over 206,704 tokens of general text, in one run.
  - Band: values strictly between the low and high pole get the neuron's mean. Random control: as many neurons per token, at random.
  - Loss change and KL(intact || ablated) on 102,998 held-out text positions. No prompt KL.
  - Sweep: layers k to 35 ablated, k = 1 to 35.

## Runs

All on a Colab L4.

```
python scripts/poles.py --config experiments/goodfire-sparsity-qwen3/poles_qwen3.yaml              # 2073s
python scripts/ablation.py --config experiments/goodfire-sparsity-qwen3/ablation_full.yaml         # 205s
python scripts/ablation.py --config experiments/goodfire-sparsity-qwen3/ablation_full_p5.yaml      # 204s
python scripts/ablation.py --config experiments/goodfire-sparsity-qwen3/ablation_sweep.yaml        # 1704s
python scripts/ablation.py --config experiments/goodfire-sparsity-qwen3/ablation_sweep_p5.yaml     # 1700s
python experiments/goodfire-sparsity-qwen3/plot.py                                                 # report.html, CPU
```

Outputs:

- `poles_qwen3/`: `thresholds.safetensors` (`p3`, `p5`, `p95`, `p97`, `mean` as [layer, neuron], fp32, and `layers`), `meta.json`.
  - Load with `emotion_neurons.ablation.load_poles(path, low, high, device)`, e.g. `low="p5", high="p95"`.
- `ablation_full/`, `ablation_full_p5/`, `ablation_sweep/`, `ablation_sweep_p5/`: `ablation.json`, one record per ablation.
- `report.html`: loss change and KL against the first ablated layer, both bands, the random control on the KL chart.

## Results: all 36 layers

Intact loss 2.187 on the held-out text.

p3 / p97:

| ablation | cells ablated | loss | loss change | KL per position | KL total |
|---|---|---|---|---|---|
| band L0-35 | 95.4% | 2.187 -> 7.037 | +4.850 | 5.327 | 548,699 |
| random L0-35 | 92.6% | 2.187 -> 13.851 | +11.664 | 12.000 | 1,235,987 |

p5 / p95:

| ablation | cells ablated | loss | loss change | KL per position | KL total |
|---|---|---|---|---|---|
| band L0-35 | 92.6% | 2.187 -> 5.191 | +3.004 | 3.442 | 354,486 |
| random L0-35 | 90.7% | 2.187 -> 13.601 | +11.414 | 11.759 | 1,211,194 |

- The tails still carry more than random neurons, but by less than on Qwen-1.
  - Derived: the band costs 2.4x (p3/p97) and 3.8x (p5/p95) less loss than random; on Qwen-1 it was 5.1x and 10.3x.
- Qwen3-8B is much less sparse than Goodfire's MoE and than Qwen-1.
  - p5/p95 here: +3.004 nats, KL 3.442. Derived: 33x Goodfire's loss change and 21x their KL, 3.7x Qwen-1's +0.812.

## Results: sweep

The band ablation of layers k to 35 only. The first row is the full ablation above.

| layers ablated | p3/p97 cells | p3/p97 loss change | p3/p97 KL | p5/p95 cells | p5/p95 loss change | p5/p95 KL |
|---|---|---|---|---|---|---|
| 0 to 35 | 95.4% | +4.850 | 5.327 | 92.6% | +3.004 | 3.442 |
| 1 to 35 | 94.8% | +2.967 | 3.316 | 91.5% | +1.366 | 1.630 |
| 2 to 35 | 94.6% | +2.620 | 2.941 | 91.3% | +1.186 | 1.426 |
| 3 to 35 | 94.4% | +2.361 | 2.664 | 91.0% | +1.046 | 1.271 |
| 4 to 35 | 94.2% | +2.158 | 2.445 | 90.8% | +0.942 | 1.155 |
| 5 to 35 | 94.1% | +1.895 | 2.165 | 90.6% | +0.792 | 0.991 |
| 6 to 35 | 93.9% | +1.611 | 1.859 | 90.4% | +0.657 | 0.839 |
| 7 to 35 | 93.8% | +1.479 | 1.715 | 90.4% | +0.601 | 0.778 |
| 8 to 35 | 93.8% | +1.425 | 1.654 | 90.4% | +0.581 | 0.752 |
| 9 to 35 | 93.9% | +1.387 | 1.603 | 90.5% | +0.564 | 0.728 |
| 10 to 35 | 94.0% | +1.333 | 1.539 | 90.6% | +0.542 | 0.696 |
| 11 to 35 | 94.2% | +1.306 | 1.506 | 90.8% | +0.528 | 0.675 |
| 12 to 35 | 94.4% | +1.267 | 1.458 | 91.1% | +0.510 | 0.650 |
| 13 to 35 | 94.7% | +1.256 | 1.441 | 91.3% | +0.499 | 0.631 |
| 14 to 35 | 94.9% | +1.222 | 1.397 | 91.6% | +0.483 | 0.608 |
| 15 to 35 | 95.2% | +1.208 | 1.375 | 91.9% | +0.478 | 0.595 |
| 16 to 35 | 95.5% | +1.153 | 1.309 | 92.1% | +0.457 | 0.567 |
| 17 to 35 | 95.7% | +1.067 | 1.208 | 92.3% | +0.431 | 0.529 |
| 18 to 35 | 95.9% | +1.005 | 1.134 | 92.6% | +0.409 | 0.501 |
| 19 to 35 | 96.1% | +0.863 | 0.975 | 92.8% | +0.365 | 0.448 |
| 20 to 35 | 96.2% | +0.704 | 0.796 | 92.8% | +0.311 | 0.385 |
| 21 to 35 | 96.2% | +0.615 | 0.695 | 92.9% | +0.278 | 0.344 |
| 22 to 35 | 96.2% | +0.503 | 0.569 | 92.8% | +0.242 | 0.299 |
| 23 to 35 | 96.1% | +0.383 | 0.433 | 92.7% | +0.194 | 0.242 |
| 24 to 35 | 95.9% | +0.298 | 0.333 | 92.5% | +0.158 | 0.199 |
| 25 to 35 | 95.8% | +0.252 | 0.281 | 92.3% | +0.137 | 0.172 |
| 26 to 35 | 95.7% | +0.215 | 0.237 | 92.2% | +0.119 | 0.149 |
| 27 to 35 | 95.6% | +0.191 | 0.206 | 92.1% | +0.108 | 0.132 |
| 28 to 35 | 95.5% | +0.169 | 0.176 | 92.0% | +0.098 | 0.115 |
| 29 to 35 | 95.4% | +0.142 | 0.146 | 91.8% | +0.082 | 0.097 |
| 30 to 35 | 95.3% | +0.127 | 0.125 | 91.7% | +0.073 | 0.085 |
| 31 to 35 | 95.2% | +0.113 | 0.108 | 91.5% | +0.065 | 0.076 |
| 32 to 35 | 95.0% | +0.102 | 0.096 | 91.3% | +0.060 | 0.068 |
| 33 to 35 | 94.8% | +0.094 | 0.085 | 91.1% | +0.059 | 0.064 |
| 34 to 35 | 94.5% | +0.075 | 0.073 | 90.6% | +0.049 | 0.059 |
| 35 to 35 | 94.1% | +0.032 | 0.062 | 90.1% | +0.021 | 0.057 |

- Sparing layer 0's MLP removes much of the damage: it drops the loss change from +4.850 to +2.967 (p3/p97) and from +3.004 to +1.366 (p5/p95).
  - Derived: 1.883 and 1.638 nats. On Qwen-1, sparing layer 0 changed almost nothing (+1.644 to +1.638).
- After layer 0, the cost falls steadily with depth, as on Qwen-1. Derived, per layer:
  - layers 1 to 19: about 0.117 (p3/p97) and 0.056 (p5/p95) ((2.967 - 0.863) / 18, (1.366 - 0.365) / 18);
  - layers 20 to 34: about 0.045 and 0.019 ((0.704 - 0.032) / 15, (0.311 - 0.021) / 15).
- Late layers come close to Goodfire's whole-model numbers: layers 24 to 35 at p5/p95 cost +0.158 with KL 0.199.
- The last layer is cheap here: layer 35 alone costs +0.032 (p3/p97) and +0.021 (p5/p95). On Qwen-1 the last layer alone was the most costly single step.

## Conclusion

- On Qwen3-8B the Goodfire claim holds only for the late layers: from layer 24 on, ablating everything between p5 and p95 costs about as much as Goodfire's whole MoE.
- Ablating the whole network costs 3 to 5 nats. Sparing only the first MLP removes 39% (p3/p97) to 55% (p5/p95) of that (derived).
- Deep layers are sparser, as on Qwen-1; the first and last layers behave differently on the two models.
