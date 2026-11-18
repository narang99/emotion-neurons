# Attention heads into MLP neurons, and a vocabulary read of the part each neuron uses

Date: 2026-11-19. Branch: `stack/04-read-rows` (PR #5). Tag: `exp/head_reads`. Status: kept.

Config: `configs/head_reads.yaml`. Results: `results/head_reads/head_reads.json`.

```
python scripts/head_reads.py --config configs/head_reads.yaml   # L4, 8.5 min for 9 neurons
```

## Question

- Which attention heads make an MLP neuron fire?
- What part of each head's write does the neuron use, and what does that part mean?

## Method

- A head h of layer k writes `w_h(t) = W_O^h z_h(t)` into the stream.
  - `z_h` is the head's 128 dims of the o_proj input.
- A neuron reads one lane (gate or up) as `read . resid_mid / rms`, with `read = row * g` (g: norm gain).
- The push of the head on the lane is `z_h(t) . r_h / rms(t)`, with `r_h = (W_O^h)^T read`. This is exact.
  - A test shows that the heads of a layer add up to that layer's attention part of the lane input.
- The neuron uses only the part of `z_h` along `r_h`.
  - In the stream, that part points along `u_h = W_O^h r_h / |r_h|`.
  - `u_h` does not change when the 128 dims of the head are rotated. Thus it is a property of the model.
  - A test shows that the used part gives all of the head's push.
- Geva's read of the used part: the logit lens of `u_h`, with the sign of the head's push.
  - For comparison, we also read the head's whole write, averaged over the tokens.
- Neurons:
  - 5 known neurons: 3068 (pride), 11109 (racing heart), 2862 (laughter), 6691 and 6612 (readers of 3068 and 11109).
  - 4 random neurons of layer 25.
- Tokens: the top 100 activations of each neuron over 340 texts (60 adult texts, 280 held-out stories).
- Ranking: first-order effect of each head on the activation, `silu'(g) u * push_gate + silu(g) * push_up`.
- Check: set the outputs of the top 3 and top 10 heads to zero at those tokens. Compare with random heads.

## What we did, step by step

1. We wrote `circuits/heads.py` and 2 tests on a small random Qwen3. Both pass.
2. We wrote `head_reads.py` and ran it on an L4 (8.5 minutes).
3. The GPU memory became full once, in the vocabulary read. The run continued and completed.
4. We fixed the memory use after the run (see "Mistakes"). We did not run the fixed code again.

## Results

- **The used part is often clear, and different from the whole write.**
  - 3068 (pride), head L22h5:
    - used part: 自豪 (pride), " proudly", " proud", " brag", " pride", 骄傲 (pride);
    - whole write: " celebr", 感动 (moved), " celebration", " emotional".
    - The head writes general emotion. The neuron takes only the pride part.
  - 3068, head L23h0: used part 归属 (belonging), 自豪 (pride), 名誉 and 荣誉 (honour), 身份 (identity).
  - 2862 (laughter), head L25h26: used part " laughing", " laughter", 笑声 (laughter), " comedic".
  - 6612 (pulse), head L23h10: used part 心血管 (cardiovascular), " heartbeat", 血压 (blood pressure).
  - 6612, head L26h31:
    - used part 体温 (body temperature), " temperature";
    - whole write 心跳 (heartbeat), " heartbeat", " pulse".
- **One head serves many neurons, each with a different direction.**
  - Head L23h0 is a top head for 3068, 11109, 6311 and 663. Its used part reads differently for each:
    - 3068: belonging, pride, honour;
    - 11109: 发热 (fever), 触发 (trigger), 情绪 (emotion), " arousal";
    - 6311 (random): 注销 (log out), " logout", " disappearance", " shutdown";
    - 663 (random): " boredom", 闲 (idle), " idle", 休息 (rest).
- **Random neurons also give clear reads.**
  - 4242: heads L23h10, L24h19, L23h11 read 入住 (check in), " dorm", 客房 (guest room), " housing", " bedrooms".
  - 663: heads read boredom, idle, " patience", " waiting".
  - Thus the method is not specific to emotion neurons.
- **Heads that push down often have no clear read.** Examples: L23h9 for 11109 (泥 (mud), " VIN", " Repair"); L23h1 for several neurons.
- **Checks** (activation change summed over 100 tokens):

  | neuron | zero top 3 | prediction | random 3 | zero top 10 | prediction | random 10 |
  |---|---|---|---|---|---|---|
  | 3068 | -40.1 | -54.4 | -0.5 | -120.5 | -156.1 | +14.8 |
  | 11109 | +118.0 | +139.9 | -14.5 | -175.2 | -241.2 | -9.4 |
  | 2862 | -450.3 | -534.8 | -40.0 | -325.5 | -445.5 | -36.9 |
  | 6691 | -15.8 | +110.8 | -9.9 | -65.3 | -69.3 | +23.2 |
  | 6612 | +227.4 | +146.5 | +5.8 | -11.6 | -180.5 | -5.1 |
  | 6311 (random) | -15.1 | -17.4 | -0.6 | -7.7 | +2.9 | -2.2 |
  | 6890 (random) | +11.3 | +17.1 | -4.6 | -7.5 | -9.7 | -2.7 |
  | 663 (random) | +121.7 | +170.3 | -58.3 | -127.0 | -138.8 | +34.1 |
  | 4242 (random) | -224.7 | -272.4 | +0.1 | -457.9 | -657.1 | +6.2 |

  - In 14 of 18 checks, the measured change has the sign of the prediction and is 0.6-1.0 of its size.
  - The other 4 checks:
    - 6691, top 3: the sign is wrong (+110.8 predicted, -15.8 measured);
    - 6612, top 10: almost no change (-180.5 predicted, -11.6 measured);
    - 6612, top 3: the sign is correct, but the change is 1.55 times the prediction;
    - 6311, top 10: the sign is wrong, but the change is small (+2.9 predicted, -7.7 measured).
  - The top heads move the neuron much more than random heads, except for 663 (random 3 heads: -58.3).

## What worked

- The exact head split. It adds up to the attention part of each layer (test).
- The used direction `u_h`. It does not depend on a basis, and its read is often clear.
- Reading the used part, not the whole write. It shows what the neuron takes from a head.

## What did not work, and limits

- The first-order prediction fails for some neurons and sizes (6691, 6612). Then the ranking is not reliable.
- Removing 3 or 10 heads at once is a large change. Heads interact through later layers, and the first-order model does not include this.
- The logit lens is weaker before about two thirds of the model. These heads are at layers 22-27, where it works acceptably.
- We do not know which source tokens each head reads from. That needs the attention pattern (next step in the checklist).
- The clear reads of random neurons show that a clear read alone does not prove a role for emotion.

## Mistakes

- **GPU memory.** The vocabulary read made an fp32 copy of the unembedding (2.5 GB) at each call.
  - The memory became full once. The run continued.
  - Fix after the run: read the vocabulary in the model's own precision, and cast `W_O` one layer at a time.
  - The results come from commit `9c57d1d`, before the fix. The fix changes only memory use and the precision of the vocabulary read.
