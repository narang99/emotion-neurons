# Local circuits of two emotion neurons: 3068 ("pride") and 11109 ("racing heart")

Run 2026-11-19. Configs `configs/neuron_circuit_{3068,11109}.yaml`. Outputs: `results/neuron_circuit_{3068,11109}/`.

```
python scripts/neuron_circuit.py --config configs/neuron_circuit_3068.yaml    # L4, 1.5 min
python scripts/neuron_circuit.py --config configs/neuron_circuit_11109.yaml   # L4, 3.5 min
```

## Setting

- Why local: whole-network circuits failed the mean-ablation check (`circuits_l25.md`), because holding every other
  neuron of 26 layers at its mean is far outside what a first-order ranking can predict.
- Neuron: layer 25. Texts: the 60 adult texts and the 280 held-out stories.
- Hits: tokens past the first 4 where the neuron is at or above its p99.9 on general text (from the top-64 harvest).
  - At p97, 3068 fires on 3819 tokens in 324 texts: its broad low firing on stories, not its pride firing.
- Writers: the neuron's gate and up inputs split exactly into the embeddings, attention per layer and each earlier
  neuron (`circuits/neuron.py`, tested to rebuild the input).
  - Ranked by first-order effect on its activation, against each writer's mean.
  - Check: hold the top writers at their mean at the hit tokens.
- Readers: every neuron of layers 26-35, by the first-order change its activation gets from this neuron's write.
  - Check: hold this neuron at its mean; a right prediction has the opposite sign of the held change.
- Readout: next-token log-probability change with the neuron held at its mean at the hit tokens.
- Values (writers', readers') are read through the final norm and the unembedding.

## Log

- **3068 ("pride"): 98 hits in 38 texts.** A pride circuit, from self-conscious-emotion writers to pride and posture
  readers.
  - Hits: "so careful, so [proud]", "He felt a [pride] so large", "puff[ing] [up] his chest"; 21 of the top 40 in
    proud stories, 11 in loving.
  - Input: the split rebuilds the gate input exactly (269.4 / 269.4); 41% from attention, 59% from MLP neurons.
  - Writers (value tokens):
    - L24 n2010: " pride", " proud", 骄傲, " proudly";
    - L22 n7269: " embarrassment", " embarrassed", 羞 (shy), " humiliation";
    - L8 n7265: 羞, 害羞 (shy), 耻 (shame), 自豪 (pride);
    - L8 n4826: 自豪, 骄傲, 成就感 (sense of achievement), " proud", " achievements".
  - Writers check: holding the top 5 changes the activation by -57.8 (first-order -84.3); 5 random writers -0.1.
    - Top 20: -124.5 (-221.8); top 100: -142.1 (-257.7). Out of 646.5 summed over the hits.
  - Readers (predicted per hit, held per hit, value tokens):
    - L26 n6691: +2.12, -1.92, " pride", 骄傲, " proud", 自豪;
    - L28 n2211: +0.76, -0.88, 骄傲, " pride", 自信 (confident), " boast";
    - L26 n4902: +0.72, -0.64, 挺 (straighten), " posture", " erect", " tall".
  - Readout: holding 3068 lowers " pride" by 0.85 nats per hit, 骄傲 0.79, 自豪 0.74, " proud" 0.67.
- **11109 ("racing heart"): 692 hits in 151 texts.** A bodily-anxiety circuit, from heartbeat and fear writers to
  pulse and unease readers.
  - Hits: "I can't sleep, [my] heart races", "he felt sick. [Sweat] dripped", "his hands [had] gone cold",
    "My hands [get] clammy"; mostly in afraid and nervous texts.
  - Input: rebuilt exactly (2099.8 / 2099.7); 59% from attention, 41% from MLP neurons.
  - Writers:
    - L22 n8705: " suddenly", 心头 (in the heart), 心跳 (heartbeat);
    - L13 n2950: 情绪 (emotion), 感情 (feeling);
    - L23 n3524: " anxiety", 恐惧 (fear), "Fear", 害怕 (afraid).
  - Writers check: top 5 held -1077.8 (first-order -2171.3); random +0.6. Out of 5748 summed over the hits.
  - Readers:
    - L27 n6612: +0.99, -0.72, " pulse", 脉 (pulse), " heartbeat", 血压 (blood pressure);
    - L27 n3729: +1.18, -1.05, 寒 (chill), " discomfort", 不安 (uneasy).
  - Readout: holding 11109 lowers " adrenal" (0.34 nats), " cortisol", " physiological", " adrenaline", " stress",
    " palms", " butterflies", 出汗 (sweating) (0.22-0.30).
- **How far the checks go:**
  - Writers: first-order predictions overshoot the measured change by 1.5-2.1x, but the top writers move the neuron
    300-1000x more than random ones.
  - Readers: the named ones (6691, 2211, 4902, 6612, 3729) match their held change in size, with opposite sign.
    - Not all top readers do: L34 n851 (predicted +0.55, held -0.03), L35 n7539 (+0.49, -0.34), and L35 n1919
      (-0.60, held -0.79, the same sign) are mispredicted.
    - The last layers' readers are the least reliable, plausibly because the final norm amplifies there.

## Surprises

- **Shame feeds pride.** Embarrassment and shyness neurons (L22 n7269, L8 n7265) are among 3068's top writers.
  - The self-conscious emotions share upstream features.
- **Pride is read as posture.** A reader (L26 n4902) writes "posture", "erect", "tall".
  - 3068 itself fires on "puffing up his chest".
- **Some top readers are not interpretable through the unembedding.** For example L30 n11405 (pred -3.35 per hit)
  writes nothing readable. Those need their own modes, not a logit-lens read.
