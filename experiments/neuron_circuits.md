# Local circuits of two emotion neurons: 3068 ("pride") and 11109 ("racing heart")

Date: 2026-11-19. Branch: `stack/03-circuits` (PR #4). Tag: `exp/neuron_circuits`. Status: kept.

Configs: `configs/neuron_circuit_3068.yaml`, `configs/neuron_circuit_11109.yaml`.
Results: `results/neuron_circuit_3068/`, `results/neuron_circuit_11109/` (`neuron_circuit.json`).

```
python scripts/neuron_circuit.py --config configs/neuron_circuit_3068.yaml    # L4, 1.5 min
python scripts/neuron_circuit.py --config configs/neuron_circuit_11109.yaml   # L4, 3.5 min
```

## Question

- For one neuron, which earlier parts make it fire?
- Which later neurons does it drive, and what do they write?
- Can we check each link?

## Why a local circuit

- The whole-network circuits failed the check (`circuits_l25.md`).
- A local circuit uses exact splits and small, local checks. A small check stays near what the model usually sees.

## Method

- Neuron: layer 25. Texts: the 60 adult texts and the 280 held-out stories.
- Hit tokens: tokens after the first 4 where the neuron is at or above its p99.9.
  - The p99.9 comes from the general-text harvest (`top64_l25`).
- **Writers.** A neuron reads `x_hat = g * resid_mid / rms` through two rows (gate and up).
  - We kept `1 / rms` fixed at each token. Then each input is linear in the stream.
  - Thus each input splits exactly into the embeddings, each layer's attention, and each earlier neuron.
  - A test shows that the parts add up to the input.
  - We ranked the writers by their first-order effect on the activation, against each writer's mean.
  - Check: we held the top writers at their mean at the hit tokens, and measured the activation.
  - Control: random writers of the same number.
- **Readers.** A later neuron reads this neuron's write in the same way. Its share is also exact.
  - We ranked the readers by the first-order change of their activation, against this neuron's mean.
  - Check: we held this neuron at its mean and measured each reader.
  - A correct prediction has the opposite sign of the held change. The prediction is for "neuron on", the check is for "neuron off".
- **Readout.** With the neuron held at its mean, we measured the change of the next-token log-probabilities.
- **Values.** We read each writer's and reader's output column through the final norm and the unembedding.

## What we did, step by step

1. We wrote `circuits/neuron.py` and 2 tests on a small random Qwen3.
   - Test 1: the split rebuilds the input exactly.
   - Test 2: a reader's share equals its own split.
2. Before the first run, we read the script again and found 3 errors (see "Mistakes"). We fixed them.
3. First run on 3068 at p97: 3819 hits in 324 texts. The script stopped in the readout (see "Mistakes").
4. We changed to p99.9 and fixed the readout. 3068 then had 98 hits in 38 texts.
5. We fixed the label of the reader numbers (see "Mistakes"), added 11109, and ran both.

## Results: 3068 ("pride")

- Its own value writes " pride", 自豪 (pride), " proud", 骄傲 (pride).
- **Hits (98 tokens in 38 texts):**
  - "so careful, so [proud]";
  - "He felt a [pride] so large";
  - "puff[ing] [up] his chest".
  - Of the top 40 hits, 21 are in proud stories and 11 in loving stories.
- **Input:** the split rebuilds the gate input exactly (269.4 and 269.4).
  - Attention gives 41%. The MLP neurons give 59%.
- **Writers** (layer, neuron, value tokens):
  - L24 n2010: " pride", " proud", 骄傲, " proudly";
  - L22 n7269: " embarrassment", " embarrassed", 羞 (shy), " humiliation";
  - L8 n7265: 羞 (shy), 害羞 (shy), 耻 (shame), 自豪 (pride);
  - L8 n4826: 自豪 (pride), 骄傲 (pride), 成就感 (sense of achievement), " proud", " achievements".
- **Writers check** (activation change summed over the hits; the clean sum is 646.5):

  | writers held | measured | first-order prediction | random writers |
  |---|---|---|---|
  | top 5 | -57.8 | -84.3 | -0.1 |
  | top 20 | -124.5 | -221.8 | -1.4 |
  | top 100 | -142.1 | -257.7 | -0.1 |

- **Readers** (predicted change per hit, change per hit when 3068 is held, value tokens):
  - L26 n6691: +2.12, -1.92, " pride", 骄傲, " proud", 自豪;
  - L28 n2211: +0.76, -0.88, 骄傲 (pride), " pride", 自信 (confident), " boast";
  - L26 n4902: +0.72, -0.64, 挺 (stand straight), " posture", " erect", " tall".
- **Readout:** with 3068 held, the log-probability per hit decreases:
  - " pride" by 0.85, 骄傲 by 0.79, 自豪 by 0.74, " proud" by 0.67.

## Results: 11109 ("racing heart")

- Its own value writes " cortisol", 生理 (physiology), " adrenal", " stress", 心跳 (heartbeat).
- **Hits (692 tokens in 151 texts):**
  - "I can't sleep, [my] heart races";
  - "he felt sick. [Sweat] dripped";
  - "his hands [had] gone cold";
  - "My hands [get] clammy".
  - Most hits are in afraid and nervous texts.
- **Input:** rebuilt exactly (2099.8 and 2099.7). Attention gives 59%, MLP neurons 41%.
- **Writers:**
  - L22 n8705: " suddenly", 心头 (in the heart), 心跳 (heartbeat);
  - L13 n2950: 情绪 (emotion), 感情 (feeling);
  - L23 n3524: " anxiety", 恐惧 (fear), "Fear", 害怕 (afraid).
- **Writers check** (the clean sum is 5748.0):

  | writers held | measured | first-order prediction | random writers |
  |---|---|---|---|
  | top 5 | -1077.8 | -2171.3 | +0.6 |
  | top 20 | -1488.4 | -3188.1 | -1.3 |
  | top 100 | -1770.6 | -3546.6 | -4.2 |

- **Readers:**
  - L27 n6612: +0.99, -0.72, " pulse", 脉 (pulse), " heartbeat", 血压 (blood pressure);
  - L27 n3729: +1.18, -1.05, 寒 (chill), " discomfort", 不安 (uneasy).
- **Readout:** with 11109 held, the log-probability per hit decreases by 0.22-0.34 for:
  - " adrenal", " cortisol", " physiological", " adrenaline", " stress", " palms", " butterflies", 出汗 (sweat).

## What worked

- The exact input split. It rebuilds the input of both neurons to 0.1 or better.
- The writers check. The top writers move the neuron 300-1000 times more than random writers.
- The named readers. Their predicted change agrees with the measured change in size and sign.
- The readout. Holding one neuron lowers the words its value writes.

## What did not work, and limits

- **First-order predictions are too large.** For writers, they are 1.5-2.1 times the measured change.
- **Some top readers are wrong.**
  - L34 n851: predicted +0.55, held -0.03.
  - L35 n7539: predicted +0.49, held -0.34.
  - L35 n1919: predicted -0.60, held -0.79 (the same sign, thus wrong).
  - The readers of the last layers are the least reliable.
- **Some top readers have no clear value.** For example, L30 n11405 (predicted -3.35 per hit).
  - The unembedding does not show what they do. They need their own modes.
- **Attention is one part.** 41% (3068) and 59% (11109) of the input come from attention. We do not know which heads.

## Mistakes

- **Mistake 1, found before the run: the writers' baseline was zero.**
  - The check holds writers at their mean. Thus the ranking must use the mean too.
  - This is the same error as round 1 of `circuits_l25.md`. We fixed it before the first run.
- **Mistake 2, found before the run: the readers used an average share.**
  - The script used the share averaged over the tokens, not the share at each token. We fixed it.
- **Mistake 3, found before the run: the readout hooks stayed on.**
  - The hooks that record the readers stayed on during the clean pass.
  - Thus the clean pass wrote over the values of the held pass. We fixed it.
- **Mistake 4, found in the run: the vocabulary size.**
  - The tokenizer has 151669 tokens. The model gives 151936 logits (padded).
  - The script used the tokenizer size and stopped. Fix: use the logit size.
- **Mistake 5, found in the run: p97 was too low for 3068.**
  - At p97, 3068 fired on 3819 tokens in 324 texts. This is its weak, broad activity on stories.
  - Fix: use p99.9. Then it fired on 98 tokens, mostly the word "proud" and its contexts.
- **Mistake 6, found in the results: the reader numbers were totals, not values per hit.**
  - The label said "per hit". The numbers were sums over 98 hits.
  - Fix: divide by the number of hits. We ran both neurons again.

## Surprises

- **Shame feeds pride.** Embarrassment and shyness neurons (L22 n7269, L8 n7265) are strong writers of 3068.
  - The self-conscious emotions share earlier features.
- **Pride is read as posture.** A reader (L26 n4902) writes "posture", "erect", "tall".
  - 3068 itself fires on "puffing up his chest".
- **The body of fear:** 11109 reads heartbeat and fear, and drives pulse and unease. Holding it lowers "adrenaline" and "sweat".
