# Goodfire sparsity on Qwen-1 7B Chat

Checks the sparsity claim of Goodfire's "Neurons as an interpretable basis" (issue #10) on a dense model.

- Claim: mean-ablating every MLP neuron value between its low and high percentile, in all layers at once, barely hurts the model.
  - Their numbers (Qwen3.6-35B-A3B, routed neurons, 8.4M web tokens, Table 1): P5 to P95 ablates 88.8% of cells, loss +0.091, KL 0.162.
  - Their random control (random-64 kept, 87.5% ablated): loss +2.328, KL 2.511.
- Here: Qwen-1 7B Chat (the refusal model), 32 layers of 11008 neurons, no experts.

## Method

- Poles (`scripts/poles.py`): per neuron, the quantiles and the mean of `silu(gate) * up` over 206,704 tokens of general text.
  - Text: fineweb-edu, TinyStories and UltraChat, 70k tokens each, 256-token windows, first 4 tokens of each window left out.
  - One forward pass per layer, stopped at that layer's `down_proj`.
- Ablation (`scripts/ablation.py`, `emotion_neurons.ablation`):
  - band: at every token, a value strictly between the low and high pole is replaced by the neuron's mean;
  - random control: at every token, as many neurons as the band ablates there, chosen at random, get their means;
  - the hooks replace the input of each ablated layer's `down_proj`; later layers see the ablated stream.
- Metrics, against the intact model on the same tokens:
  - held-out general text: the next 35k tokens per corpus after the poles' windows (102,998 scored positions);
  - loss: mean next-token cross-entropy, intact and ablated, and the change;
  - KL(intact || ablated) of the next-token distribution, per position and summed over all positions.
- `ablation.json` also holds KL at the post-instruction positions of the refusal val prompts. It is not used here.
  - Five template tokens per prompt are a weak measure of refusal; the refusal rate under ablation is in `todo.md`.
- Sweep: the band ablation of layers k to 31 only, for k = 1 to 31; layers before k stay intact.

## Runs

All on a Colab L4.

```
python scripts/poles.py --config experiments/goodfire-sparsity/poles_qwen1.yaml          # p3, p97, mean; 1680s
python scripts/ablation.py --config experiments/goodfire-sparsity/ablation_full.yaml     # 190s
python scripts/ablation.py --config experiments/goodfire-sparsity/ablation_sweep.yaml    # 1493s
python scripts/poles.py --config experiments/goodfire-sparsity/poles_qwen1_p5.yaml       # p5, p95, mean; 1605s
python scripts/ablation.py --config experiments/goodfire-sparsity/ablation_full_p5.yaml
python scripts/ablation.py --config experiments/goodfire-sparsity/ablation_sweep_p5.yaml # 1502s
python experiments/goodfire-sparsity/plot.py                                             # report.html, CPU
```

Outputs, one folder per run id:

- `poles_qwen1/`, `poles_qwen1_p5/`: `thresholds.safetensors` (each quantile and `mean` as [layer, neuron], fp32, and `layers`), `meta.json`.
  - Load with `emotion_neurons.ablation.load_poles(path, low, high, device)`.
- `ablation_full/`, `ablation_sweep/`, `ablation_full_p5/`, `ablation_sweep_p5/`: `ablation.json`, one record per ablation.
- `report.html`: line charts of loss change and KL against the first ablated layer, from `plot.py`.
  - The p3/p97 runs wrote before the move out of `results/`, so their `thresholds` field reads `experiments/goodfire-sparsity/results/poles_qwen1/...`. It is the same file as `poles_qwen1/`.

## Results: all 32 layers

Intact loss 2.062 on the held-out text.

p3 / p97:

| ablation | cells ablated | loss | loss change | KL per position | KL total |
|---|---|---|---|---|---|
| band L0-31 | 94.9% | 2.062 -> 3.706 | +1.644 | 1.907 | 196,448 |
| random L0-31 | 91.2% | 2.062 -> 10.385 | +8.323 | 8.945 | 921,351 |


p5 / p95:

| ablation | cells ablated | loss | loss change | KL per position | KL total |
|---|---|---|---|---|---|
| band L0-31 | 91.3% | 2.062 -> 2.874 | +0.812 | 0.978 | 100,684 |
| random L0-31 | 88.5% | 2.062 -> 10.420 | +8.358 | 8.982 | 925,085 |


- The tails carry most of the function: the band costs far less than the random control at a similar count.
  - Derived: the loss change is 5.1x smaller than random at p3/p97 (1.644 vs 8.323), 10.3x at p5/p95 (0.812 vs 8.358).
- The dense model is far less sparse than Goodfire's MoE.
  - p5/p95 here ablates 91.3% of cells and costs +0.812 nats, KL 0.978. Goodfire's P5 to P95 ablates 88.8% and costs +0.091, KL 0.162.
  - Derived: about 9x the loss change and 6x the KL at a similar band.
- The random control ablates fewer cells than the band (88.5% to 91.2%).
  - Its count per token comes from the band test on its own disturbed activations, which sit outside the band more often.
  - Per layer it ranges from 72.8% to 98.5%, so it is a slightly gentler control. It is still far worse.

## Results: sweep, p3 / p97

The band ablation of layers k to 31 only. The first row is the full ablation above.

| layers ablated | cells ablated | loss change | KL per position |
|---|---|---|---|
| 0 to 31 | 94.9% | +1.644 | 1.907 |
| 1 to 31 | 95.1% | +1.638 | 1.904 |
| 2 to 31 | 95.0% | +1.453 | 1.705 |
| 3 to 31 | 94.6% | +1.279 | 1.514 |
| 4 to 31 | 94.5% | +1.197 | 1.418 |
| 5 to 31 | 94.4% | +1.128 | 1.337 |
| 6 to 31 | 94.4% | +1.095 | 1.298 |
| 7 to 31 | 94.5% | +1.065 | 1.266 |
| 8 to 31 | 94.4% | +1.000 | 1.192 |
| 9 to 31 | 94.5% | +0.974 | 1.167 |
| 10 to 31 | 94.6% | +0.954 | 1.137 |
| 11 to 31 | 94.7% | +0.903 | 1.072 |
| 12 to 31 | 94.8% | +0.846 | 1.008 |
| 13 to 31 | 94.8% | +0.778 | 0.921 |
| 14 to 31 | 94.8% | +0.705 | 0.823 |
| 15 to 31 | 94.9% | +0.639 | 0.742 |
| 16 to 31 | 95.1% | +0.574 | 0.656 |
| 17 to 31 | 95.1% | +0.501 | 0.567 |
| 18 to 31 | 95.2% | +0.451 | 0.492 |
| 19 to 31 | 95.1% | +0.409 | 0.436 |
| 20 to 31 | 95.1% | +0.372 | 0.390 |
| 21 to 31 | 95.1% | +0.340 | 0.346 |
| 22 to 31 | 95.0% | +0.316 | 0.310 |
| 23 to 31 | 94.9% | +0.292 | 0.282 |
| 24 to 31 | 94.8% | +0.277 | 0.259 |
| 25 to 31 | 94.7% | +0.260 | 0.238 |
| 26 to 31 | 94.6% | +0.250 | 0.225 |
| 27 to 31 | 94.5% | +0.243 | 0.216 |
| 28 to 31 | 94.4% | +0.234 | 0.206 |
| 29 to 31 | 94.2% | +0.226 | 0.196 |
| 30 to 31 | 94.1% | +0.210 | 0.182 |
| 31 to 31 | 94.1% | +0.155 | 0.134 |


- Late layers tolerate the ablation: layers 24 to 31 cost +0.277, layer 31 alone +0.155.
- Most of the damage is in layers 2 to 17: starting at 2 costs +1.453, starting at 18 costs +0.451.
- Layers 0 and 1 add almost nothing: +1.644 with them, +1.638 without.

## Results: sweep, p5 / p95

The band ablation of layers k to 31 only. The first row is the full ablation above.

| layers ablated | cells ablated | loss change | KL per position |
|---|---|---|---|
| 0 to 31 | 91.3% | +0.812 | 0.978 |
| 1 to 31 | 91.5% | +0.755 | 0.918 |
| 2 to 31 | 91.2% | +0.647 | 0.794 |
| 3 to 31 | 90.8% | +0.550 | 0.690 |
| 4 to 31 | 90.7% | +0.504 | 0.641 |
| 5 to 31 | 90.6% | +0.483 | 0.614 |
| 6 to 31 | 90.6% | +0.474 | 0.600 |
| 7 to 31 | 90.7% | +0.464 | 0.590 |
| 8 to 31 | 90.6% | +0.435 | 0.561 |
| 9 to 31 | 90.7% | +0.420 | 0.547 |
| 10 to 31 | 90.9% | +0.412 | 0.529 |
| 11 to 31 | 91.0% | +0.397 | 0.506 |
| 12 to 31 | 91.0% | +0.376 | 0.477 |
| 13 to 31 | 91.1% | +0.357 | 0.447 |
| 14 to 31 | 91.1% | +0.336 | 0.413 |
| 15 to 31 | 91.2% | +0.319 | 0.384 |
| 16 to 31 | 91.4% | +0.299 | 0.354 |
| 17 to 31 | 91.4% | +0.277 | 0.319 |
| 18 to 31 | 91.5% | +0.262 | 0.287 |
| 19 to 31 | 91.4% | +0.242 | 0.259 |
| 20 to 31 | 91.4% | +0.225 | 0.234 |
| 21 to 31 | 91.3% | +0.208 | 0.210 |
| 22 to 31 | 91.3% | +0.196 | 0.190 |
| 23 to 31 | 91.1% | +0.183 | 0.173 |
| 24 to 31 | 91.1% | +0.174 | 0.160 |
| 25 to 31 | 90.9% | +0.165 | 0.147 |
| 26 to 31 | 90.7% | +0.159 | 0.140 |
| 27 to 31 | 90.6% | +0.155 | 0.135 |
| 28 to 31 | 90.5% | +0.150 | 0.129 |
| 29 to 31 | 90.3% | +0.145 | 0.123 |
| 30 to 31 | 90.1% | +0.136 | 0.116 |
| 31 to 31 | 90.1% | +0.102 | 0.087 |


- The same shape as p3/p97, at about half the cost.
- Late layers alone come close to Goodfire's numbers: layers 24 to 31 cost +0.174 with KL 0.160.
  - Goodfire's whole model at P5 to P95: +0.091, KL 0.162.
- Layer 31 alone costs +0.102, more than Goodfire's whole model at the same band.

## Conclusion

- The tails carry most of the function on this dense model too: the band costs 5x (p3/p97) to 10x (p5/p95) less than random.
- The "barely hurts" part does not hold for the whole network: +0.81 nats at p5/p95, about 9x Goodfire's MoE.
- Deep layers are sparser, as assumed. The cost of each layer, derived from the sweep:
  - p3/p97: about 0.063 per layer for layers 2 to 17 ((1.453 - 0.451) / 16), about 0.017 for layers 24 to 30 ((0.277 - 0.155) / 7);
  - p5/p95: about 0.024 per layer for layers 2 to 17 ((0.647 - 0.262) / 16), about 0.010 for layers 24 to 30 ((0.174 - 0.102) / 7).
- The last layer is the exception: alone it costs as much as 9 to 10 of layers 24 to 30 (0.155 / 0.017, 0.102 / 0.010).
