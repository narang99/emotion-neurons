# emotion-neurons

Which MLP neurons of Qwen3-8B (layer 25 of 36) build the model's emotion directions?

This branch keeps four things:

- the code that extracts activations and builds the emotion vectors;
- the code that finds each neuron's poles;
- a summary of the sparse-set methods we tried, in `experiments/SPARSE_SETS.md`;
- the greedy sets of neurons for each emotion's write (below).

Every other experiment lives at its git tag. `experiments/LOG.md` lists all of them, in order, with their results.

## Setup

```
git clone git@github.com:narang99/emotion-neurons.git && cd emotion-neurons
uv sync
bash scripts/fetch_runs.sh    # runs/stage2_l25: pooled sums and story order, 263 MB, public
```

## Pipeline

| step | command | where | output |
|---|---|---|---|
| 1. activations | `python scripts/extract.py --config configs/stage2_l25.yaml` | GPU, about 10 min on an L4 | `runs/stage2_l25/{sums.safetensors, stories.json}` |
| 2. emotion vectors | `python scripts/build_vectors.py --config configs/stage2_l25.yaml --device cpu` | CPU, seconds | `results/stage2_l25/{vectors.safetensors, vectors_meta.json, checks.json}` |
| 3. poles | `python scripts/poles.py --config configs/poles.yaml` | GPU, about 10 min on an L4 | `results/poles/{thresholds.safetensors, meta.json}` |

- Step 2 needs only step 1's output. `fetch_runs.sh` gets it, so no GPU is needed for step 2.
- Steps 2 and 3 are committed under `results/`.

### Step 1: activations

- Data: 1680 stories in `data/stories/`. 12 emotions x 20 topics x 6 stories, and 240 neutral stories.
  - Held out: story_idx 5 of each emotion (20 per emotion), neutral story_idx 10 and 11 (40).
- One pass to layer 25. Per story, the sums over tokens 0-49 and 50+ of four sites:
  - `resid_mid` (after attention), `resid_post` (the layer's output), `mlp_act` (the 12288 neuron activations);
  - `mlp_out = resid_post - resid_mid`, derived.
- The run keeps sums, not activations. Means, differences and the MLP write are linear in them.

### Step 2: emotion vectors

- Per site: the mean over an emotion's 100 train stories (tokens 50+), minus the mean over the 12 emotions.
- `denoised`: the top neutral-story PCs (50% of variance) are projected out.
- Check: nearest-vector accuracy on held-out stories, 12-way, chance 0.08.
  - `resid_post` denoised: 0.83 held-out. `mlp_act`: 0.83. `mlp_out`: 0.81.

```python
from safetensors.torch import load_file

V = load_file("results/stage2_l25/vectors.safetensors")
# keys <site>/<mean|neutral>/<raw|denoised>, each [12, dim]; emotion order in vectors_meta.json
delta = V["mlp_out/mean/raw"]  # [12, 4096]: the MLP's emotion write
dA = V["mlp_act/mean/raw"]  # [12, 12288]: per-neuron activation shift; delta_e = W_down @ dA_e
```

### Step 3: poles

- A token is at neuron n's low pole if `a_n <= p3`, and at its high pole if `a_n >= p97`.
- p3 and p97 are taken over 206,704 tokens of general text: fineweb-edu, TinyStories and UltraChat, 70k tokens each.

```python
P = load_file("results/poles/thresholds.safetensors")
P["p3"], P["p97"]  # [12288] each
```

## Greedy sets: which neurons build each emotion's write

- Tag `exp/greedy_sets_l25`, log `experiments/greedy_sets_l25.md`.
- Per emotion, the fewest layer-25 neurons that rebuild 90% of the emotion's MLP write.
  - Write: `W_down @ (mean activation on the emotion's 120 stories - mean on all 1680 stories)`. Exact, no PCA, no split.
  - Neurons outside the set sit at their all-story mean.
  - Greedy: add the neuron that rebuilds the most of what is still missing. Stop at 90%.
- Set sizes: 1472 (loving) to 2025 (calm). Happy: 1527.

### Reproduce (CPU, under a minute)

```
uv sync
bash scripts/fetch_runs.sh    # runs/stage2_l25
uv run python scripts/greedy_sets.py --config configs/greedy_sets_l25.yaml
```

- The first run downloads layer 25's `W_down` from the Hub (200 MB) into `~/.cache/huggingface/emotion-neurons/`.

### Read the results

- `results/greedy_sets_l25/sets.json`: `sets[<emotion>]` is the list of neurons, in the order greedy added them.
- `results/greedy_sets_l25/neurons/<emotion>.csv`: one row per neuron in the set.
  - `rank`: the step greedy added it.
  - `write_share_pct`: its part of the write, along the write. Over all 12288 neurons it sums to 100.
    - It can be negative: the neuron pushes against the write, but fixes a side direction.
    - Over the set it sums to about 83%, not 90%. The other neurons fill the rest of the 90% by cancelling side directions.
  - `rebuilt_after_pct`: the share of the write the set rebuilds once this neuron is in.
  - `abar_emotion`, `abar_all`, `shift`: mean activation on the emotion's stories, on all stories, and the difference.
- `results/greedy_sets_l25/tensors.safetensors`: `write`, `shift`, `abar_emotion` [12, dim] in `EMOTIONS` order, and `abar_all` [dim].
- `results/greedy_sets_l25/summary.json`: per emotion, the set size, and the size at 50%, 75%, 90% and 95%.

```python
import json
from safetensors.torch import load_file

from emotion_neurons.models import load_w_down

sets = json.load(open("results/greedy_sets_l25/sets.json"))["sets"]
T = load_file("results/greedy_sets_l25/tensors.safetensors")
W = load_w_down("Qwen/Qwen3-8B", 25)  # [4096, 12288]; column n is neuron n's write
happy = sets["happy"]  # 1527 neurons; happy[0] = 2862
part = W[:, happy] @ T["shift"][0, happy]  # what the set writes; rebuilds 90% of T["write"][0]
```

## Modes of an emotion's neurons: run it for another emotion

What each neuron of an emotion's greedy set fires on, at both poles, judged blind. Done for happy
(`results/harvest_happy_l25/report.html`, log `experiments/harvest_happy_l25.md`).

One command runs it end to end, with a checkpoint after each stage:

```
python scripts/pipeline.py --config configs/harvest_angry_l25.yaml
```

| stage | where | time (L4) | output in results/<run_id>/ |
|---|---|---|---|
| harvest | GPU | ~12 min | harvest.json (+ runs/<run_id>/, 2-3 GB) |
| modes | GPU | ~25-30 min | modes.json, modes/batch_XX.txt |
| judge | CPU, OpenRouter key | ~15 min, under $1 | judgements/<model>/batch_XX.jsonl |
| report | CPU | ~1 min | report.html, neurons.csv |

- **Restarts:** rerun the same command. Finished stages are skipped (markers in `results/<run_id>/stages/`).
  - Inside a stage, the harvest reuses a saved corpus pass and the judge skips finished batch files.
- **GPU and CPU on different machines:** without CUDA the run stops before the first GPU stage that is not done.
  - Run it on Colab with `notebooks/modes_pipeline.ipynb` (File > Open notebook > GitHub, `narang99/emotion-neurons`).
  - Then run the same command on your machine; it continues with judge and report.
  - To move the files: `--pipeline.s3-prefix s3://your-bucket/...` on both machines, or the notebook's download cell
    (unzip into `results/`).
- **Stop early:** `--pipeline.until modes`.
- **Quick check of a new setup:** `configs/pipeline_smoke_happy.yaml` (36 neurons, ~90k tokens, ~5 min on an L4).
- **OpenRouter key:** `OPENROUTER_API_KEY`, or a file at `~/.config/openrouter/key`
  (`mkdir -p ~/.config/openrouter && echo "<key>" > ~/.config/openrouter/key && chmod 600 ~/.config/openrouter/key`).
  The judges are Haiku 5.5 and Gemini 2.5 Flash-Lite (`judge.models`).

**A new emotion:** copy `configs/harvest_angry_l25.yaml` and set `run_id`, `emotion` and `harvest.neurons_csv`
(`results/greedy_sets_l25/neurons/<emotion>.csv`). The config refuses an emotion that does not match its CSV.

**Reading the report** (`report.html`, opens in a browser; `neurons.csv` has one row per neuron):

- "No match" neurons push toward the emotion but fire on something else, on the side the emotion uses.
- Check both poles. The side is read from a story-mean shift, and a few strong firings at the other pole can hide
  behind it (happy's 3832: joyful firings at its high pole, but the word "laugh" drives its mean down).
- "Match" leans on the harvest corpus: almost every happy-scene cluster came from TinyStories, the only narrative
  source. Add a fiction corpus to `harvest.corpora` for emotions that children's stories rarely show.

## Layout

| path | what |
|---|---|
| `src/emotion_neurons/` | library: `config`, `models` (load, hooks), `texts` (stories, corpora), `capture` (pooled sums), `vectors`, `poles`, `sparse` (greedy sets) |
| `scripts/` | thin scripts: parse the config, call the library, save; `pipeline.py` runs the modes report end to end |
| `configs/` | one YAML per run |
| `data/stories/` | the stories |
| `results/` | small outputs, committed |
| `runs/` | large outputs, gitignored; public copies via `scripts/fetch_runs.sh` |
| `experiments/` | `LOG.md` (every experiment, tag, result) and `SPARSE_SETS.md` |
| `tests/` | exact identities: vectors, pooled sums, quantiles |

## Parked questions

- Neurons at about p75: they help the direction but sit at no pole. Part of the feature or not?
- "A feeling was just named" neurons (48, 2066, 3582) fire after any feeling word. Are they in the emotion-specific group?
