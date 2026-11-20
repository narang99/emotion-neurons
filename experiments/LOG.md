# Experiment log

This file is the index of all experiments, in order. It also records how the work went: the method, the mistakes, what worked and what did not.

## How to read this file

- Each experiment has one row in the table.
- Each experiment has a tag at the commit that made its results.
- Each experiment has a detailed log in `experiments/`.
- Status:
  - **kept**: the result is in `main` (or in a PR for `main`) and is part of the final result;
  - **failed**: the method did not work. The branch and the tag stay. The log tells why;
  - **superseded**: a later experiment replaced it.

## Index

| tag | status | what | result | log |
|---|---|---|---|---|
| `exp/stories` | kept | 1680 stories by Claude: 12 emotions x 20 topics x 6, and 240 neutral | the validator and the audit pass | `PLAN.md` |
| `exp/stage2_l25` | kept | emotion vectors at layer 25: emotion mean minus the 12-emotion mean, neutral directions removed | held-out 12-way accuracy 0.81-0.84 | `stage2_l25.md` |
| `exp/stage3_l25` | kept | the neurons behind each MLP write; a 2M-token harvest; modes named by Haiku | the 0.9 / 0.9 rule keeps all 12288 neurons; 0.30 of the hits of the pos-pole neurons are in feeling-word modes (controls 0.15) | `stage3_l25.md` |
| `exp/min_set_l25` | superseded | greedy minimum set for each emotion write | cosine 0.9 with 700-1050 neurons; replaced by sparse sets with a held-out stop | `top64_l25.md` |
| `exp/sparse_l25` | kept | sparse sets with held-out transfer; swaps, cross-fit, shrunk targets; modes of the top 64 | 298-1233 neurons match the full MLP; no knee; hill-climbing gains are 6.5% or less | `top64_l25.md` |
| `exp/sparse_l25` (cross-fit, round 1) | failed | cross-fit with a cross-half Gram matrix | the Gram matrix was indefinite and the transfer was about 0; fixed with the halves' own norms | `top64_l25.md` |
| `exp/probe_kv_l25` | kept | the top 64 on 60 adult texts; values through the unembedding | 3068 (pride), 2862 (laughter) and 11109 (racing heart) read and write their concept; next-token agreement is at chance | `probe_kv_l25.md` |
| `exp/probe_kv_l25` (mode assignment) | failed | put new hits into modes from 5 exemplars | too strict: 2% pass; needs the full cluster centres | `probe_kv_l25.md` |
| `refactor/src-layout` | kept | the code becomes a `src/` library with thin scripts | CPU and L4 results are identical to the old code | "Refactor" below |
| `exp/journey_all` | kept | emotion vectors at all 36 layers: attention and MLP parts, direction, accuracy | accuracy is highest at layer 22 (0.87); the layer-25 vector is 66% attention and 34% MLP; the last-layer vector is 40% and 60% | `journey_all.md` |
| `exp/sparse_layers` | kept | sparse sets of the MLP emotion write at every layer | cosine 0.9 needs about 1200 neurons at layers 10-20, 570 at layers 28-34, 38 at layer 35 | `sparse_layers.md` |
| `exp/circuits_l25-round1` | failed | whole-network circuits for the layer-25 score, ranked by effect against zero | faithfulness -0.1 at 100-1000 neurons; it ranked constant activations | `circuits_l25.md` |
| `exp/circuits_l25-round2` | failed | the same, ranked by effect against the mean | faithfulness -0.04 to -0.21 at 100-3000 neurons; the check is too far from normal model states | `circuits_l25.md` |
| `exp/neuron_circuits` | kept | local circuits of 3068 (pride) and 11109 (racing heart) | shame and pride writers drive 3068; 3068 drives pride and posture readers; holding 3068 lowers " pride" by 0.85 nats | `neuron_circuits.md` |
| `exp/head_reads` | kept | attention heads into 9 MLP neurons (5 known, 4 random); logit lens of the part of each head the neuron uses | the used part is often clear and differs from the whole write (L22h5: whole write "celebration", 3068 uses "pride"); one head (L23h0) serves many neurons with different directions; 14 of 18 checks agree with the prediction | `head_reads.md` |
| `exp/causal_sets` | kept | remove or swap each emotion's shift at layer 25 on its sparse set, random sets, all neurons | the set (5.9% of neurons) gives about 70% of all neurons' effect, random sets 5-20% of it; but MLP-25 moves later emotion scores by only about 10% at natural size, and no answer flips | `causal_sets.md` |
| `exp/word_tests` | kept | per-layer emotion read on implied texts, and at negated distractor words | at "was not afraid" the vector reads "afraid" at all layers (about 0.5 in the middle layers), the true emotion 0.04-0.09 | `word_tests.md` |
| `exp/tail_check` | kept | split each emotion's shift into tokens above p97, below p3, middle | sparse-set neurons: 10% of tokens in a tail carry 75% of the set's write; all neurons: the middle carries one third | `tail_check.md` |
| `exp/harvest_layers` | kept | top tokens of early sparse-set neurons, layers 0, 2, 8, 12 | emotion-word share of top tokens 0.006-0.045 (controls 0.000-0.006): pronouns, body, posture words, not emotion words | `harvest_layers.md` |
| `exp/chain-round1` | failed | chain token -> L8 n4826 / n7265 -> n3068 | holding the L8 neurons moves n3068 like random neurons (0.04 / 0.09 vs 0.10); guessed middles | `chain.md` |
| `exp/chain` | kept | scan by patching, then the chain " proud" -> MLP0 -> L2 n8663 -> n3068 -> " pride", with L22 n3221 and L24 n2010 | n8663 patched back restores 64% (random 0.1%); n3221 and n2010 are necessary (-1.06, -0.96; 26-31x random); n3068 held: " pride" -1.01 nats | `chain.md` |
| `exp/swap_values` | kept | swap the `W_down` columns of two emotions' sets at layer 25 | gap change -1.7 to +3.7 of 55-71 (random pairs -0.3 to +0.6); much smaller than the shift swap: the emotion is in the shift, not in the values | `causal_sets.md` |
| `exp/chain_scans` | kept | patching scans for 2862 (laughter) and 11109 (racing heart) | 2862: token -> MLP 0 -> L2 n2285 (0.22, random 0.00), like 3068; 11109 (context hits): spread, no neuron above 0.04 | `chain.md` |
| `exp/input_rows` | kept | Geva's lens on gate/up rows through `W_E` and `W_U`, checked against harvest top tokens | input lens: layer 0 overlap 0.45 (shuffled 0.008), fails from layer 2; output lens works from layer 8 (0.075 vs 0.007); layer 25 reads clearly ("pride", "adrenaline", "laughter") | `input_rows.md` |
| `exp/chain_context` | kept | patching scan for n3068 at hits that are not a pride word | spread: best neuron 0.036, best 10 together 0.10; "pride" from context has no short path | `chain.md` |
| `exp/input_rows-negative` | kept | the input-row read, also by most negative activation | L2 n8663 reads " Pride, pride, 傲, 骄傲" through the input embedding | `input_rows.md` |
| `exp/tail_causal` | kept | clamp set neurons into [p3, p97] at layer 25 | true score -3.5% (set) vs -7.2% for the whole shift and -0.3% for random sets: tails are about half of the causal effect | `tail_check.md` |
| `exp/crossval` | kept | 6-fold accuracy and shrinkage kappa at layer 25 | accuracy 0.772 +- 0.048 (Stage 2's split 0.829 is the best fold); kappa 1 beats plain by 0.004 at 256 neurons, within noise | `crossval.md` |
| `exp/direction_reports` | kept | split each emotion's MLP-25 write into modes (set tail firings) and remaining; lens, top tokens, examples | modes and remaining have equal norms, cosine 0.25-0.39; modes reads clearly (hug, heartbeat, pride), remaining does not, but peaks on other cues of the same scenes | `direction_reports.md` |
| `exp/raw_sets` | kept | per token, greedy on raw `a_n(t) w_n` (no mean removed) toward the L25 emotion vector, at high-score tokens; where members sit in their range | sets at cos 0.5: 10% of members at a pole (all neurons 7%, random-direction sets 0.2%), carrying 23% of the projection; a random direction is fit with about the same k (285 vs 277), so per-token sets are not unique; emotion sets differ from random ones by moderate displacement (p75-p97 0.32 vs 0.14) | `raw_sets.md` |

## Record of the session of 2026-11-18 and 2026-11-19

### Questions from the review, and what we found

- An external reviewer said that the MLP write is "mostly not the emotion vector". We tested this.
  - The MLP write has cosine 0.32-0.41 with the layer-25 emotion vector.
  - But the part of the write at right angles to all 12 vectors still gives 0.73 held-out accuracy (chance: 0.08).
  - The part along the vectors gives 0.69. Thus most of the write is emotion information, but not along the vector.
  - Topic cannot cause it: each emotion uses the same 20 topics, 6 stories each.
- **Mistake:** before this test, I agreed with the reviewer and said that most neurons "serve directions with no relation to the emotion". The test showed that this was wrong. I corrected it.
- With neutral directions removed or not, the cosine changes only a little (0.41 raw, 0.37 denoised).
- Conclusion: by the paper's recipe, the MLP write is exactly the MLP's part of the emotion vector. No extra projection is necessary.

### Refactor

- **What we did:**
  - We moved the code into a `src/emotion_neurons/` library: models, texts, capture, vectors, decomposition, harvest, modes, values, circuits.
  - The vector recipe is isolated behind `EmotionVectors`.
  - The scripts became thin: they read the config, call the library, and save.
  - We added `jaxtyping`, `einops`, ruff and a pre-commit hook (from in-tfm).
  - We added tests for exact identities.
- **How we checked it:**
  - First, we copied all old results to a baseline folder.
  - On CPU, we ran each refactored script and compared each output with `scripts/compare_results.py`.
  - On the L4, we ran extraction, vectors, projection, report, harvest, modes and the adult-text probe again.
  - All results are identical, or differ only by float rounding (1e-16) or by the order of two equal values.
- **Mistakes and problems during the refactor:**
  - The write tool did not replace an existing file that I had not read. The first comparison thus used the old file and said "identical". I saw that the script had failed, replaced the file, and compared again.
  - A zsh loop did not split `$c` into words. Thus two scripts did not run, and the comparison used old files. I ran each script on its own line.
  - The new package needed `einops` and `jaxtyping` on its branch. The first run failed until I added them.
  - The committed `modes_summary.md` of Stage 3 was old. The old script on `main` also makes the new version. Thus the refactor did not cause the difference.
  - `neuron_set.json` changed the order of two dimensions with exactly equal values (an unstable sort). The sets are the same.

### Work method

- I ran GPU work on a Colab L4 as a background process, with a log file and an exit line.
- I synced each result to S3 and then downloaded it.
- I compared each new result with an earlier result where one existed (layer 25 in each new experiment).
- I wrote each experiment as a branch in a stack of PRs (`gh stack`): #2, #3, #4.
- I kept failed experiments with a tag and a log.

### What worked

- Exact splits: the stream is a sum, so each part is exact. Tests show this for each split.
- Regression checks against earlier results. They found no error in the refactor.
- Local circuits with local checks. The top writers move a neuron 300-1000 times more than random writers.

### What did not work

- Whole-network circuits with a first-order ranking (two rounds). The mean-ablation check is too far from normal model states.
- Mode assignment of new tokens from 5 exemplars. It is too strict.
- `k_match` in layers 12-19. Many emotions never reach the ceiling there.

### Process mistakes, and what I changed

- **Colab connection:** the browser connection stopped once. I could not release the VM. The user released it.
- **Silent stale code:** a failed `git checkout` did not stop the run script, and old code ran. Now each git step is on its own line, and the script checks the branch and the code.
- **Out of memory:** the backward pass made gradients for all weights. Now weights have no gradients.
- **Force-push:** the tool did not let me force-push the rebased stack. I pushed the stack under new names (`stack/0N-...`). The old `exp/0N-...` branches on GitHub are stale. Later log changes go up the stack by merge, not by rebase.
- **`gh pr edit`:** it fails with a GitHub "Projects (classic)" error. I used `gh api -X PATCH .../pulls/N`.
- **Too early claims:** two times I gave a result before I checked it (the reviewer point, and round 2 of the circuits). Each time, I found the error and corrected it. I now check the run log and the numbers before I report.
