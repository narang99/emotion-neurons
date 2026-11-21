# Handoff for an outside opinion (2026-11-21)

A self-contained summary of what this project does, what we found, and where we are unsure.
Written so that a reader with no context can give an opinion. Branch: `stack/06-raw-sets` (PR #8, stacked on #7).

## 1. The project in one paragraph

- Model: Qwen3-8B. Focus: the MLP of layer 25 (of 36), 12288 neurons.
- We build "emotion vectors" the way recent papers do: the mean residual stream on stories of one emotion, minus a baseline.
  - 12 emotions (happy, inspired, loving, proud, calm, desperate, angry, guilty, sad, afraid, nervous, surprised), plus neutral.
  - 100 train and 20 held-out stories per emotion (200 + 40 neutral). Means over tokens 50+ of each story.
- Main question: **how do individual MLP neurons build an emotion's direction?**
  - Two views compete.
    - Goodfire-style: neurons are features; a neuron contributes when it fires strongly (its "modes", the tails of its distribution).
    - Mean-vector style: the direction is a difference of means; small shifts on many neurons give the same mean.

## 2. Notation

- `a_n(t)`: activation of neuron n at token t (input to `down_proj`). `w_n`: column n of `W_down` (4096-d).
- MLP output at t: `sum_n a_n(t) w_n`. Exact.
- An emotion's MLP write: `d = W_down (abar_own - abar_base)`, with `abar` the story-mean activations averaged over stories.
  It splits exactly over neurons: `d = sum_n u_n`, `u_n = (abar_own[n] - abar_base[n]) w_n`.
- Poles: activation <= p3 or >= p97 of the neuron's own distribution on 207k tokens of general text.
- Mean ablation: keep a set S at its emotion values, hold every other neuron at its mean. The write is then `sum_{n in S} u_n`.

## 3. What we tested, in order (one line each)

| # | question | answer | log |
|---|---|---|---|
| 1 | Do emotion vectors exist, where? | Yes, from mid layers; layer-25 held-out 12-way accuracy about 0.8 (6-fold 0.77 +- 0.05) | `stage2_l25.md`, `crossval.md` |
| 2 | Can a small neuron set reproduce the MLP's emotion direction? (method 1) | 268 to 1401 neurons (mean 728, 6%) match the whole MLP's held-out cosine | `stage3_l25.md` |
| 3 | Does it matter causally? | The set gives ~70% of the whole MLP's effect (random sets 5-20%); but the whole MLP-25 moves later emotion scores only ~10%; no answer flips | `causal_sets.md` |
| 4 | Tails or small shifts? | In the sets, pole tokens (~10% of tokens) carry ~75% of the write; clamping the tails gives half the effect of removing the whole shift | `tail_check.md` |
| 5 | Raw activations, no baseline: which neurons have zero, constant or emotion-specific mean push? | 31% zero (cancel), 57% constant offset (syntax neurons, on any text), 13% emotion-specific | `signal_noise.md` |
| 6 | Fewest neurons that rebuild the write in direction AND length (method 2) | ~1900 neurons (12-18%); the rest cancel as a vector but still add 11-23% along the emotion axis | `residual_sets.md` |
| 7 | Do method 2's extra neurons look emotional? (blind Haiku judges) | Same as method 1's: 21% vs 21% (random 0%); ~37% look like syntax | `judge_sets.md` |
| 8 | One traced circuit | " proud" token -> MLP0 -> L2 n8663 -> L25 n3068 -> " pride" logit, with random baselines | `chain.md` |
| 9 | Does the vector track the story's emotion or the word? | Mostly the word: at "not angry" the vector reads angry; the model's own answer stays right | `word_tests.md` |

One-page summary of rows 5 to 7: `raw_activations_summary.md`.

## 4. The two set searches (rows 2 and 6)

Both grow a set from empty, greedily, one neuron per step. Exact updates via the Gram matrix `(W^T W) * (shift shift^T)`.

| | method 1 (match sets) | method 2 (signal sets) |
|---|---|---|
| baseline in `d` | mean over the 12 emotions | mean over all other stories (11 emotions + neutral) |
| neutral directions removed from `W` | yes (top PCs of neutral stories, 50% variance) | no |
| score | cosine `cos(sum_S u_n, d)` | leftover `|d - sum_S u_n| / |d|` (direction and length) |
| stop | held-out cosine reaches the whole MLP's held-out cosine | leftover reaches the train target's own sampling noise (0.24-0.37) |
| size | 268-1401 | 1499-2232 |

- Plot: `results/residual_sets/set_curves.png` (log x) and `set_curves_linear.png`.
- **The two searches trace almost the same curve of cosine vs set size.** They differ only in where they stop.
- Cosine rises about linearly in log(set size) up to ~2000 neurons. No elbow, no small core.
- The match set is 78-92% inside the signal set.

## 5. Main conclusions so far

1. An emotion's layer-25 MLP write is spread over ~2000 neurons. There is no small core.
2. About 70% of the emotion-specific part comes from pole firings, on ~8% of tokens. Every method gives about this number.
3. Over half of all neurons add a constant offset (syntax, punctuation, pronouns). It is removed by any baseline.
4. Only about 1 in 5 of the neurons that build the direction looks emotion-related from its top contexts.
   The related ones are situations and body signs (fleeing, chest tight, eerie sounds), rather than emotion words.
5. Causally, MLP-25 is a small part of how the model answers.

## 6. Where we are unsure (opinions wanted)

1. **Is layer-25 MLP the right object?** It moves later scores ~10% and flips no answers. Its write has cosine only
   ~0.35 with the layer-25 residual emotion vector. Should we first find where the answer comes from (attention, other
   layers) with a patching scan on the answer logits?
2. **Which baseline is right?** 12-emotion mean (method 1) vs all-other-stories mean (method 2) vs none (row 5).
   We think a baseline is needed because 57% of neurons add a constant offset, and zero-ablation is far off-distribution
   (the rest's mean write is 4-7x the emotion gap). Is mean ablation the right choice here?
3. **Are the stopping rules sound?** Method 1 stops at held-out parity; method 2 at the train noise level. Both are
   points on one smooth curve. Is there a principled "size" of a distributed feature?
4. **Are the "poles" the right unit?** Thresholds are p3/p97 on general text. The low pole of a SiLU-gated neuron
   (`silu(gate) * up`, signed) means "pushed below its usual floor", which is not the same as strong firing.
5. **Is judging by top contexts fair?** Neurons can matter through moderate firing that top contexts miss.
   One judge per neuron; no agreement check yet.
6. **The vector reads words** (row 9). Does that undercut the whole "emotion vector" framing at this layer?

## 7. Parked questions

- Neurons near p75 that help the direction but sit at no pole: part of the feature or not?
- "A feeling was just named" neurons (48, 2066, 3582): are they in the emotion-specific group?

## 8. Files

- Library: `src/emotion_neurons/` (`decomposition/` holds `sparse.py` (method 1), `residual.py` (method 2), `signal_noise.py`).
- Scripts: `scripts/` (thin). Configs: `configs/`. Small results: `results/` (committed). Large: `runs/` and S3.
- Logs: `experiments/<name>.md`, index `experiments/LOG.md` (every experiment, tag, status, one-line result).
- Every experiment has a git tag `exp/<name>` at the commit that produced it. Superseded work (`exp/raw_sets`,
  `exp/raw_pools`: per-token cosine sets, which fit a random direction as well as the emotion) lives only at its tag.
- Reproduce CPU-only results from the cache `runs/signal_noise/story_classes.safetensors` (on S3):
  `python scripts/residual_sets.py --config configs/residual_sets.yaml --device cpu` (30 s).
