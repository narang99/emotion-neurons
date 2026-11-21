# emotion-neurons

Which MLP neurons of Qwen3-8B (layer 25 of 36) build the model's emotion directions?

This branch keeps only three things:

- the code that extracts activations and builds the emotion vectors;
- the code that finds each neuron's poles;
- a summary of the sparse-set methods we tried, in `experiments/SPARSE_SETS.md`.

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
delta = V["mlp_out/mean/raw"]   # [12, 4096]: the MLP's emotion write
dA = V["mlp_act/mean/raw"]      # [12, 12288]: per-neuron activation shift; delta_e = W_down @ dA_e
```

### Step 3: poles

- A token is at neuron n's low pole if `a_n <= p3`, and at its high pole if `a_n >= p97`.
- p3 and p97 are taken over 206,704 tokens of general text: fineweb-edu, TinyStories and UltraChat, 70k tokens each.

```python
P = load_file("results/poles/thresholds.safetensors")
P["p3"], P["p97"]               # [12288] each
```

## Layout

| path | what |
|---|---|
| `src/emotion_neurons/` | library: `config`, `models` (load, hooks), `texts` (stories, corpora), `capture` (pooled sums), `vectors`, `poles` |
| `scripts/` | thin scripts: parse the config, call the library, save |
| `configs/` | one YAML per run |
| `data/stories/` | the stories |
| `results/` | small outputs, committed |
| `runs/` | large outputs, gitignored; public copies via `scripts/fetch_runs.sh` |
| `experiments/` | `LOG.md` (every experiment, tag, result) and `SPARSE_SETS.md` |
| `tests/` | exact identities: vectors, pooled sums, quantiles |

## Parked questions

- Neurons at about p75: they help the direction but sit at no pole. Part of the feature or not?
- "A feeling was just named" neurons (48, 2066, 3582) fire after any feeling word. Are they in the emotion-specific group?
