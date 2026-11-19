# Handoff: next run (2026-11-19)

## Current state (latest)

- Done, with results and logs: causal sets, value swap, word tests, tail check, early-layer harvest, chain (3068),
  chain scans (2862, 11109, 3068 context hits), input rows (both signs). See `LOG.md`.
- **Not run yet.** These 3 jobs were started, but the VM was released on request before any of them finished. S3 has no results for them. Run all 3:

  | run | command | what |
  |---|---|---|
  | `tail_causal` | `python scripts/tail_causal.py --config configs/tail_causal.yaml` | clamp set neurons into [p3, p97] vs random sets (GPU, ~12 min) |
  | `direction_reports` | `python scripts/direction_reports.py --config configs/direction_reports.yaml` | main / modes / remaining directions: lens, top tokens, example sentences (GPU, ~10 min) |
  | `crossval` | `python scripts/crossval.py --config configs/crossval.yaml --device cpu` | 6-fold accuracy and kappa (CPU, needs `bash scripts/fetch_runs.sh stage2_l25`) |

- To restart:
  1. Connect Colab on an L4, and run the setup cell (secrets, clone, editable install).
  2. Run the 3 jobs. `direction_reports` and `tail_causal` share the GPU, so run them one after the other. `crossval` (CPU) can run beside them.
     - Run one git step per line, and never launch two jobs at the same moment: they race on the git lock.
  3. Download the results, write the logs (`experiments/<run>.md`, STE), add LOG rows and tags, and tick the README checklist.
- Known fixes in this round:
  - `crossval.py` must run with `--device cpu`. The greedy `SetState` makes its tensors on the CPU.

## Earlier state (kept for the record)

**Update, end of 2026-11-19: steps 0-5 are done.** See `LOG.md` rows `causal_sets` to `chain`. The open items are in the
README checklist. The text below is the state before the runs, kept for the record.

Branch: `stack/05-causal-sets` (on top of PR #6). Goal: results with a random baseline, and the full chain
token -> early neuron -> L25 n3068 -> " pride" logit.

## State

- **Causal sets (step 0): code done, run not done.**
  - Code: `src/emotion_neurons/interventions.py`, `scripts/causal_sets.py`, `configs/causal_sets.yaml`.
  - Tests: `tests/test_interventions.py` (2 pass).
  - The first run on the L4 reached 80 of 240 stories. The VM was then released. No results were saved.
  - Speed: about 1.1 s per story for 10 conditions. Estimated total: about 10 minutes.
- **Word tests (step 1): code done, never run.**
  - Code: `scripts/word_tests.py`, `configs/word_tests.yaml`.
  - Texts: `data/word_tests/mislead.jsonl` (24 texts, 2 per emotion, written by Claude).
    - Each has a distractor emotion word 3 or more times. The target word is absent (checked by a script).
  - It also uses the 24 implied texts in `data/adult_stories` (explicit = false). 1 of them has a label word.
  - Note: only 14 of the 240 held-out training stories contain one of the 12 label words.
  - Estimated run time: about 5 minutes.
- **Steps 2 to 5: not started.** See "Plan" below.

## How to start the next run

1. Connect Colab and switch to an L4.
2. Setup cell: secrets `HF_TOKEN`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`; clone over HTTPS;
   `uv pip install --system --no-deps -e .`, then `pydantic-settings einops jaxtyping awscli`; assert `+cu` and CUDA.
3. Run script (one git step per line, with a branch check), detached with `Popen`, log with `EXIT`:

   ```
   git fetch -q origin
   git checkout -q stack/05-causal-sets
   git reset -q --hard origin/stack/05-causal-sets
   test "$(git branch --show-current)" = "stack/05-causal-sets"
   uv pip install --system -q --no-deps -e .
   python scripts/causal_sets.py --config configs/causal_sets.yaml
   python scripts/word_tests.py --config configs/word_tests.yaml
   aws s3 sync results/causal_sets s3://narang99-private/emotion-neurons/causal_sets/results
   aws s3 sync runs/causal_sets s3://narang99-private/emotion-neurons/causal_sets/runs
   aws s3 sync results/word_tests s3://narang99-private/emotion-neurons/word_tests/results
   ```

4. Check: `EXIT 0`, the `[total]` line, the JSON files exist. Then download and read.
5. Release the VM with `runtime.unassign()`.

## Plan

0. **Causal sets at layer 25.**
   - On each held-out story of emotion e:
     - remove its shift `dA_e` from its sparse set, from 3 random sets of the same size, and from all neurons;
     - mean-ablate the sparse set and the random sets.
   - Swaps (happy/sad, calm/afraid, proud/guilty, both ways, scales 1 and 4):
     - add `scale * (dA_f - dA_e)` on the union of the two sets, on 3 random sets, and on all neurons.
   - Readouts:
     - emotion scores at layers 25, 28, 31, 35;
     - the model's one-word answer to "how does the main character feel?";
     - a few continuations.
1. **Words vs situations.**
   - The question: at which layer does the vector follow the situation, not the word?
   - Readouts: per-layer accuracy, and the share of texts labelled as the distractor.
2. **Early-layer modes, cached.**
   - Harvest layers 0, 8 and 12 in one pass: the sparse sets from `results/sparse_layers/`, plus random controls.
   - Save per layer to `runs/harvest_layers/L<k>/` and to S3. Skip a layer whose files exist.
   - Name modes by top tokens first. Use Haiku only for unclear neurons, and only if the user agrees.
3. **Links from early to late neurons.**
   - Use the writer edges of `neuron_circuit`, from 3068 down to layer 8, and then from layer 8 to layers 0-2.
4. **Trace to input tokens.**
   - Split an early neuron's input exactly into three parts:
     - the embedding at its own position;
     - each head, split by source token: `z_h(t) = sum_s A(t,s) v_h(s)`, with attention frozen;
     - earlier MLPs.
   - Also read its input rows against `W_E` (checklist item 2).
5. **The demo chain.**
   - Chain: " proud" / 自豪 -> head or embedding -> L8 n4826 -> L25 n3068 -> " pride".
   - Check each link by ablation, against random neurons and random heads.

## Rules to keep

- Subagents only when the user asks (the auto-mode check refused a Haiku subagent for step 1).
- Each experiment: a stacked PR, an `exp/<name>` tag, and a line in `experiments/LOG.md` (STE).
- Release the VM when done.
