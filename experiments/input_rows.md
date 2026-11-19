# Geva's lens on MLP input rows: what a neuron listens for, from weights alone

Date: 2026-11-19. Branch: `stack/05-causal-sets`. Tag: `exp/input_rows`. Status: kept.

Config: `configs/input_rows.yaml`. Results: `results/input_rows/input_rows.json`.

```
python scripts/input_rows.py --config configs/input_rows.yaml   # Colab CPU (no forward pass), a few minutes
```

## Question

- Can we read what a neuron responds to from its gate and up rows alone, as Geva et al. read values?
- Does the read agree with what the neuron really fires on?

## Method

- A neuron of layer l computes `a = silu(W_gate[n] . (g * x/rms)) * (W_up[n] . (g * x/rms))`.
- Two lenses put a token's vector in place of x and compute `a` for all 151,669 tokens:
  - `embed`: x = the input embedding `W_E[tok]`;
  - `unembed`: x = `g_final * W_U[tok]`, the stream direction that predicts the token.
- Check:
  - For the harvested neurons of layers 0, 2, 8 and 12 (80 to 89 per layer, controls included), take their 10 most frequent top-activation tokens.
  - Measure the share of those tokens that are in the lens' top 200.
  - Baseline: the lens of another neuron of the same layer.

## Results

- **Agreement with the harvest:**

  | layer | embed lens | shuffled | neurons with any match | unembed lens | shuffled | neurons with any match |
  |---|---|---|---|---|---|---|
  | 0 | **0.447** | 0.008 | 0.82 | 0.008 | 0.002 | 0.08 |
  | 2 | 0.029 | 0.003 | 0.10 | 0.004 | 0.005 | 0.04 |
  | 8 | 0.004 | 0.000 | 0.05 | **0.075** | 0.007 | 0.35 |
  | 12 | 0.004 | 0.001 | 0.04 | 0.027 | 0.001 | 0.16 |

  - The input lens predicts layer-0 neurons well. From layer 2 on, the stream at a token is no longer close to its embedding.
  - The output lens starts to work at layer 8. It is 10 times the shuffled baseline there, but it finds only 7.5% of the tokens.
- **Reads of the chain and circuit neurons (output lens):**
  - L25 n3068: 自豪 (pride), " pride", " proudly", " proud", 骄傲 (pride), 责任感 (sense of responsibility), 成就感 (sense of achievement).
  - L25 n11109: " adrenaline", " cortisol", " stress", " anxiety", " arousal".
  - L25 n2862: " laughter", " laughing", " gigg", " laughed", 笑声 (laughter).
  - L24 n2010: " offspring", 儿子 (son), 孩子 (child), " grandchildren", 欣慰 (gratified), " approving", " applaud". It listens for pride in one's children.
  - L8 n4826: " triumph", 自豪 (pride), " achievements", 胜利 (victory), 成就感 (sense of achievement), " brag".
  - L8 n7265: " pride", 自豪 (pride), " embarrassment", " timid".
  - L22 n3221 and L2 n8663: no clear read.

## Conclusion

- The input rows can be read, but each lens works only at some depths:
  - the input embedding at layer 0;
  - the output unembedding from about layer 8, and clearly at layers 24-25.
- At layer 25 the read agrees with the neuron's concept and with its value read. 3068, 11109 and 2862 read and write the same concept.

## Limits

- The lens ranks tokens by positive activation. A neuron that fires negatively, such as L2 n8663 on " proud", cannot be read this way. The next step is to rank by |a|, or by `a` of the right sign.
- The lens ignores attention and the earlier layers. At layer 2, neither lens matches the harvest.
