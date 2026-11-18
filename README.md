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

Open ideas, untested: cross-validated held-out (5 folds instead of one 20-story split); modes for the sparse "match"
sets; tuning the shrinkage kappa by cross-validation; other layers (rerun the pipeline with a new config).

## Layout

| path | what |
|---|---|
| `data/stories/` | 1440 emotional + 240 neutral stories (Stage 1), committed |
| `configs/*.yaml` | one per run; everything that distinguishes runs lives here |
| `emotion_neurons/` | config models (`config.py`), ranged safetensors read of `W_down` (`weights.py`) |
| `scripts/` | one script per step; each docstring has its command |
| `results/<run>/` | small outputs, committed |
| `runs/<run>/` | large outputs, gitignored (see below) |

## Running

```
uv sync
uv run python scripts/sparse_set.py --config configs/sparse_l25.yaml --sparse-set.target raw --run-id sparse_l25_raw
uv run --with matplotlib python scripts/plot_sparse.py --config configs/sparse_l25.yaml --run-id sparse_l25_raw
```

- CPU scripts (`neuron_set`, `min_set`, `sparse_set`, `plot_sparse`, `report*`, `summarize_modes`) run locally. They
  fetch only layer 25's `down_proj` from the Hugging Face shard (~100 MB download, cached as float32 under `$HF_HOME/emotion-neurons`).
- GPU scripts (`extract`, `project`, `harvest`, `modes`) ran on a Colab L4 (bf16, no quantization).
- Large inputs are not in git:
  - `runs/stage2_l25/{sums.safetensors,stories.json}` (264 MB): pooled per-story activations; needed by `neuron_set`,
    `min_set`, `sparse_set`. Regenerate with `scripts/extract.py --config configs/stage2_l25.yaml` on an L4.
  - `runs/top64_l25/{window.npy,windows.json}`: harvest windows; needed by `report_neurons`. Regenerate with
    `scripts/harvest.py --config configs/top64_l25.yaml`.
- Bulk subagent work (stories, mode naming) uses Haiku; see `CLAUDE.md`.
