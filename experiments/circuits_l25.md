# Whole-network circuits for the layer-25 emotion score (failed)

Date: 2026-11-19. Branch: `stack/03-circuits` (PR #4).
Tags: `exp/circuits_l25-round1`, `exp/circuits_l25-round2`. Status: failed.

Config: `configs/circuits_l25.yaml`. Results: `results/circuits_l25/` (round 2).

```
python scripts/layer_vectors.py --config configs/layers_all.yaml    # CPU, once
python scripts/circuits.py      --config configs/circuits_l25.yaml  # L4, 3 min for 60 stories
```

## Question

- Can a small set of neurons, in all layers up to 25, keep the emotion score of a story?
- Which emotions' sets do those neurons belong to?

## Method

- Target: the emotion score of each token, `s_e(t) = u_e . resid_post_25(t)`.
  - `u_e` is the denoised layer-25 vector of the story's emotion.
  - We used the tokens from 50 on.
- Stories: 60 held-out stories, 5 for each emotion.
- Effect of each neuron in layers 0-25: one backward pass.
  - We froze the attention outputs. Thus attention is one fixed part, and all paths stay at the same token.
  - We froze the scale of each RMSNorm. Thus each norm is a linear map at that token.
  - SwiGLU stays as it is. Thus the effect is first-order.
- Direct parts: neuron write, attention and embeddings along `u_e`. They add up to the score exactly.
- Circuit: the top k neurons by the size of their effect.
- Check (faithfulness): `(s(C) - s(none)) / (s(all) - s(none))`.
  - `s(C)`: the score when all neurons outside the circuit C are held at their 12-emotion mean.
  - `s(none)`: all neurons of layers 0-25 held at their mean.
  - `s(all)`: the model without change.
  - Control: random sets of the same size.

## What we did, step by step

1. We wrote the tool (`circuits/attribution.py`, `edges.py`, `graph.py`) and 5 tests on a small random Qwen3.
   - The tests show that the direct parts add up to the score, and the edges are exact.
2. First L4 run: the GPU memory was full in the backward pass.
   - Cause: the model weights required gradients. The backward pass made gradients for 8B weights.
   - Fix: no gradients for weights. Attention runs without autograd, because its output is frozen.
3. Round 1 ran. The circuits did not keep the score (see Results).
4. We found the cause and changed the ranking (round 2).
5. The first round-2 run used the old code by mistake (see "Mistakes"). We ran it again correctly.
6. Round 2 also failed. We stopped this method and used local circuits (`neuron_circuits.md`).

## Results

- **The direct parts add up to the score** with a maximum error of 0.8% (bf16).
- **Round 1: effect against zero activation (`a * ds/da`). Failed.**

  | circuit size | faithfulness | random set |
  |---|---|---|
  | 100 | -0.10 | 0.01 |
  | 1000 | -0.12 | -0.02 |
  | 3000 | 0.03 | 0.01 |
  | 10000 | 0.24 | 0.08 |
  | 30000 | 0.53 | 0.04 |

  - Cause: the effect counts all of a neuron's activation, also the part that is the same for all emotions.
  - Layer 0 had 29% of the effect of the top 1000 neurons. These neurons have large, almost constant activations.
  - Mean ablation almost does not change these neurons. Thus, keeping them does not help.
  - The sum of all neuron effects (median 8882) was 6 times the change that ablation can make (1539).
- **Round 2: effect against the mean activation (`(a - mean) * ds/da`). Failed.**

  | circuit size | faithfulness | random set |
  |---|---|---|
  | 100 | -0.04 | 0.01 |
  | 300 | -0.12 | 0.00 |
  | 1000 | -0.21 | -0.02 |
  | 3000 | -0.13 | 0.01 |
  | 10000 | 0.17 | 0.08 |
  | 30000 | 0.75 | 0.04 |

  - Layer 0 still had 33% of the effect of the top 1000 neurons.

## Why the method failed

- The check holds all other neurons of 26 layers at their mean at the same time.
- This puts the stream far from what the model usually sees.
- The changes go through attention and the norms. The attribution froze both.
- A first-order, same-token ranking cannot predict this large change.

## What worked

- The exact parts: the direct parts add up to the score, and the input edges are exact. Tests show this.
- The tests on a small random model found no error in the maths.

## What did not work, and mistakes

- **The method did not work.** Both rankings failed the check.
- **Mistake: out of GPU memory.** The weights required gradients. We turned them off.
- **Mistake: the first round-2 run used the old code.**
  - `git checkout` failed because old result files were in the way.
  - The run script used `cmd1 && cmd2`. With `set -e`, bash does not stop on a failure in such a chain.
  - Thus the old code ran, and it gave the same numbers as round 1.
  - I first told the user that round 2 failed. That was wrong at that time, because round 2 did not run.
  - I saw the identical numbers, checked the log, and found the cause.
  - Fix: the run script now does each git step on its own line, checks the branch, and checks the code.
- **Mistake: in round 1, the result tables used a ranking that did not match the check.** The ranking used zero as the baseline, but the check used the mean.
- **The emotion-set split of these circuits has no meaning,** because the circuits are not faithful.
  - Round 2: 16% of the effect in the story's own emotion set, 35% in other sets, 50% in no set.

## Next

- Local circuits around one neuron, from exact splits, with local checks. See `neuron_circuits.md`.
