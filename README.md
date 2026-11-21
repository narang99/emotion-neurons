# emotion-neurons

Which MLP neurons of Qwen3-8B (layer 25 of 36) write the model's emotion directions, and what do they fire on?

- Plan and fixed choices: `PLAN.md`.
- One log per experiment in `experiments/` (setting, rounds, results, surprises). Read them in order:
  1. `stage2_l25.md`: emotion vectors from 1440 stories (12 emotions); held-out 12-way accuracy 0.81–0.84.
  2. `stage3_l25.md`: neurons behind each emotion's MLP write; p97+ harvest over 2M tokens; HDBSCAN modes named by
     Haiku.
  3. `top64_l25.md`: sparse neuron sets per emotion write, and what the top 64 do. Latest.

## Where things stand (2026-11-18)

- The MLP's emotion write splits exactly over neurons: `delta_e = sum_n W_down[:, n] * dA_e[n]`.
- **Sparse sets** (`results/sparse_l25_raw/sets.json`): per emotion, 298 (happy) to 1233 (angry) of 12288 neurons
  generalise to held-out stories as well as the full MLP; 700–1050 keep train cos 0.9. The 12 sets share neurons: 4772
  distinct in total, 977 in 3+ emotions.
- **No small core:** cos rises linearly in log k (`results/sparse_l25_raw/tradeoff.png`). Swaps, cross-fitting and
  shrunk targets gain <= 6.5%, within held-out noise.
- **Top 64** (`results/top64_l25/neurons_report.md`): more emotion-related than random neurons (27% of p97+ hits in
  emotion modes vs 16%), but mostly emotion-general (fire around any feeling word). One clean single-emotion neuron:
  3068 ("proud").

## Checklist

- [ ] **Attention into an emotion neuron, by head** (GPU, L4, short pass; start with 3068, layers 23-25).
  - For one reader neuron, a head pushes it along one direction only: `r_h = W_O^h^T read`. Thus the minimal set of
    attention directions is at most one direction for each head. This does not depend on a basis.
  - Rank the heads by their push on the reader's gate and up input at its hit tokens. Find the minimal set of heads.
  - Check: set the write of the top heads to zero at the hits. Compare with random heads.
  - Read each top head's write, projected onto `r_h`, through the unembedding (Geva on the part the neuron uses).
  - Do the same heads appear for all neurons of one emotion set at one layer? Show the distribution of head scores.
  - Do not give meaning to single `o_proj` input dimensions. Inside a head they can be rotated without changing
    the model. If a per-dimension claim is necessary, first compare the minimal set in the standard basis with
    random rotations and with the SVD basis of `W_O^h W_V^h`.
- [ ] **Geva on what an MLP neuron reads** (CPU): read each neuron's gate and up rows through the unembedding.
  - The input-side read of Dar, Geva et al. (2023). It is clearer in late layers than in early layers.
  - Compare with its modes and with its value read.
- [ ] **Trace a neuron back to the input** (GPU, L4, about 15-20 min): recursive writers of 2862 (happy #1),
  3 writers for each step, down to layer 0. Then split attention by head and source token, so the trace reaches tokens.
- [x] **Non-children texts probe** (done, `experiments/probe_kv_l25.md`) (GPU, L4): do the top-64 neurons hit p97, and which modes, on 60 new emotional
  texts? Ready: `scripts/probe_stories.py`, `configs/adult_probe_l25.yaml`, `data/adult_stories/`.
- [x] **Key-value read** (per neuron done; whole-write read partly) (CPU, Geva et al. 2021): what each neuron's value promotes through the unembedding, and
  whether it predicts the next token at its key's hits. `scripts/kv_values.py`, `configs/kv_values_l25.yaml`.
  Add the same read for each emotion's whole write and its sparse set's write: per-neuron values that read as
  noise but a clear combined write would favour a distributed code (Hazy Research, "Stuffing MLPs Full of Facts").
- [x] **Swap happy and sad values** (done, `experiments/causal_sets.md`: the activation-shift swap and the `W_down` column swap) (GPU, causal): at layer 25, swap the `W_down` columns of happy's sparse set with
  sad's (pair by greedy rank; the sets overlap, so leave shared neurons alone), or inject sad's activation shift
  into happy's set on happy stories. Read out: projections on the emotion vectors at later layers, and the
  continuation the model writes. A flip towards sad means the set's values carry the emotion.
- [x] **Tail check** (done, `experiments/tail_check.md`): per-token activations of each sparse set on the stories, to see whether an emotion's shift
  comes from tokens above p97 / below p3 or from a small move on every token.
- [x] **Trace to the input** (done for n3068, `experiments/chain.md`): token -> MLP 0 -> L2 n8663 -> n3068 -> " pride", by an exact
  source split (`circuits/sources.py`) and a patching scan (`scripts/chain_scan.py`).
- [x] **Geva on MLP input rows** (done, `experiments/input_rows.md`) (CPU): read `W_gate[n] * g` and `W_up[n] * g` through `W_U` and `W_E`, for traced and random neurons.
- [x] **More chains** (done for 11109 and 2862, `experiments/chain.md`).
- [x] A chain for non-lexical hits (n3068 at context hits, `experiments/chain.md`) and an input read of negative firing (`experiments/input_rows.md`).
- [x] **Directions of the write** (done, `experiments/direction_reports.md`) (`scripts/direction_reports.py`): split each emotion's MLP-25 write into `modes` (the set's
  tail firings) and `remaining`; logit lens, top tokens and example sentences for main / modes / remaining.
- [x] **Causal tail test** (done, `experiments/tail_check.md`): clamp set neurons into [p3, p97] against random sets.
- [x] Cross-validated held-out (done, `experiments/crossval.md`) (5 folds instead of one 20-story split); tuning the shrinkage kappa by cross-validation.
- [x] Other layers (done: vectors at all 36 layers, `experiments/journey_all.md`; sparse sets at all layers,
  `experiments/sparse_layers.md`; harvests at layers 0, 2, 8, 12, `experiments/harvest_layers.md`).
- [x] **Raw activations, no mean removed** (done): which neurons carry the emotion write, and do the rest cancel?
  - `experiments/signal_noise.md`: noise, offset and emotion-specific neurons; modes of the offset neurons.
  - `experiments/residual_sets.md`: the signal set that rebuilds the write, and what the rest does.
  - The per-token versions (`exp/raw_sets`, `exp/raw_pools`) are superseded and kept only at their tags.
- [ ] **Parked: neurons at about p75.** Neurons that help the direction but sit near p75, not at a pole. What do we do
  with them: count them as part of the feature, or as a separate kind of contribution?
- [ ] **Parked: "a feeling was just named" neurons.** Several top writers (48, 2066, 3582) fire after any feeling word
  (`experiments/stage3_l25.md`). Is the emotion-specific part of the write in the many small terms instead of the top 10?

## Layout

| path | what |
|---|---|
| `data/stories/` | 1440 emotional + 240 neutral stories (Stage 1), committed |
| `configs/*.yaml` | one per run; everything that distinguishes runs lives here |
| `src/emotion_neurons/` | the library (below) |
| `scripts/` | one thin script per step; each docstring has its command |
| `tests/` | exact identities: neuron terms sum to the write, greedy and swaps pick the best move |
| `results/<run>/` | small outputs, committed |
| `runs/<run>/` | large outputs, gitignored (see below) |

Library (`src/emotion_neurons/`):

| module | what |
|---|---|
| `config.py`, `emotions.py` | the typed run config; the 12 emotions and site names |
| `models/` | Qwen3 loading and hooks (`qwen3.py`); single tensors from the Hub by ranged read (`weights.py`) |
| `texts/` | story sets and the held-out split; harvest corpora |
| `capture/` | pooled per-story sums and means, single layer or all layers |
| `vectors/` | how emotion vectors are built (`recipes.py`, isolated behind `EmotionVectors`); denoising; checks |
| `decomposition/` | activation shifts and neuron terms; sparse sets (greedy, swaps, shrink); older set methods |
| `harvest.py`, `modes.py`, `values.py` | corpus activations and percentile bands; mode clustering; logit-lens reads of values |

## Picking up

```
git clone git@github.com:narang99/emotion-neurons.git && cd emotion-neurons
uv sync
bash scripts/fetch_runs.sh               # gitignored run data, ~610 MB, public, no AWS account needed
uv run python scripts/sparse_set.py --config configs/sparse_l25.yaml --sparse-set.target raw --run-id sparse_l25_raw
uv run --with matplotlib python scripts/plot_sparse.py --config configs/sparse_l25.yaml --run-id sparse_l25_raw
```

- Every small output is committed under `results/`. The large ones are in a public bucket, at the same paths under
  `https://narang-public-s3.s3.amazonaws.com/narang99-pages/emotion-neurons/runs/<run>/<file>`; `scripts/fetch_runs.sh`
  downloads them (the bucket can't be listed, so the script names each file).

  | run data | size | what | needed by |
  |---|---|---|---|
  | `runs/stage2_l25/{sums.safetensors,stories.json}` | 263 MB | per-story pooled sums of the 4 layer-25 sites, story metadata (train / held-out split) | `build_vectors`, `sparse_set` |
  | `runs/top64_l25/{acts,pos,tokens,window}.npy, windows.json` | 340 MB | 2M-token harvest of the top 64 neurons (activation memmap, token ids, positions, corpus windows) | `modes`, `report_neurons` |
  | `runs/stage3_l25/` (same files) | 1.7 GB | harvest of Stage 3's 419 ranked neurons + 50 controls; `bash scripts/fetch_runs.sh stage3_l25` | `modes` for stage3 |

- CPU scripts (`neuron_set`, `min_set`, `sparse_set`, `plot_sparse`, `report*`, `summarize_modes`) run locally. They
  fetch only layer 25's `down_proj` from the Hugging Face shard (~100 MB download, cached as float32 under
  `$HF_HOME/emotion-neurons`).
- GPU scripts (`extract`, `project`, `harvest`, `modes`) ran on a Colab L4 (bf16, no quantization); each docstring has
  its command. A new layer or story set is a new config and a rerun of `extract` onward.
- Bulk subagent work (stories, mode naming) uses Haiku; see `CLAUDE.md`.

## Using the findings

```python
import json
from safetensors.torch import load_file

# Emotion vectors: per site, mean over an emotion's 100 train stories minus the mean over the 12 emotions.
# Keys <site>/<mean|neutral>/<raw|denoised> [12, dim], sites resid_mid, resid_post, mlp_act, mlp_out;
# order = vectors_meta.json["emotions"]. "denoised" = neutral-story PCs projected out.
V = load_file("results/stage2_l25/vectors.safetensors")
delta = V["mlp_out/mean/raw"]        # [12, 4096]: the MLP's emotion write
dA = V["mlp_act/mean/raw"]           # [12, 12288]: per-neuron activation shift; delta_e = W_down @ dA_e

# Sparse neuron set per emotion (indices into the 12288 neurons, in greedy order)
sets = json.load(open("results/sparse_l25_raw/sets.json"))
sets["happy"]["match"]       # {k, neurons, cos, transfer}: smallest set that generalises as well as the full MLP
sets["happy"]["tau0.9"]      # {k, neurons, cos, transfer, ...}: smallest set found with train cos >= 0.9

from emotion_neurons.models import load_w_down
W = load_w_down("Qwen/Qwen3-8B", 25)  # [4096, 12288] float32
```

- Full greedy orders and the cos / transfer curves for every k and method: `results/sparse_l25_{raw,denoised}/sparse_set.json`.
- What each of the top 64 neurons fires on (Haiku-named modes, share of p97+ hits, corpus):
  `results/top64_l25/neurons_report.md`; the raw mode clusters are in `results/top64_l25/modes.json`, names in
  `modes/names_*.jsonl`. Same for Stage 3's ranked neurons under `results/stage3_l25/`.
