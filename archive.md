# Archive

Everything built in this repo before the cleanup of 2026-11-23 is on the branch `archive`.

- Read a file: `git show archive:<path>`.
- Each experiment also has a tag `exp/<name>` at the commit that made its results.
  - Read its log: `git show <tag>:experiments/<log>.md`.
- `experiments/LOG.md` on `archive` is the full index of experiments, with results and failures.

## Still on main

- Model adapters (`models/`): Qwen3 and other native models; Qwen-1 7B Chat converted to Qwen2.
- Datasets (`datasets/`): emotion stories and benchmark instructions as categories of examples.
- Difference of means (`vectors/diff_means.py`), neutral-PC denoising (`vectors/denoise.py`).
- Emotion vectors at one layer: `scripts/extract.py`, `scripts/build_vectors.py`, `capture/`, `vectors/`.
- The Arditi et al. direction: selection, ablation, addition, generation check (`directions.py`, `scripts/direction.py`).
- Neuron poles, p3 and p97 over general text (`poles.py`, `scripts/poles.py`).

## Only on archive

Paths are relative to `src/emotion_neurons/` or the repo root.

### Neuron sets

- Greedy sets: the fewest neurons whose shifts rebuild 90% of an emotion's MLP write.
  - `sparse/`, `scripts/greedy_sets.py`, `configs/greedy_sets_l25.yaml`, `results/greedy_sets_l25/`.
  - Tag `exp/greedy_sets_l25`.
- The neuron set behind a direction (e.g. refusal), in the greedy-set format.
  - `directions.py` (`run_direction_neurons`), `scripts/direction_neurons.py`. Tag `exp/refusal_qwen1`.
- Older set searches (relaxed masks, cross-fit, residual sets, raw sets): summary in `experiments/SPARSE_SETS.md`.

### What neurons fire on

- Harvest: the top hits of chosen neurons at both poles over general text, optionally sampled per percentile band.
  - `harvest.py`, `scripts/harvest.py`. Tags `exp/harvest_happy_l25`, `exp/harvest_bands`.
- Harvest of every neuron of one MLP into a store (built, smoke-tested on CPU, never run on a GPU).
  - `harvest_all.py`, `scripts/harvest_all.py`, `configs/harvest_all_qwen1_l17.yaml`.
  - Known bug: the process does not exit after "done".
- Modes: HDBSCAN clusters of the MLP input at a neuron's hits, per pole.
  - `modes.py`, `scripts/modes.py`.
- Blind judges of mode clusters on OpenRouter (Haiku 5.5, Gemini 2.5 Flash-Lite).
  - `judge.py`, `scripts/judge_modes.py`, `data/prompts/`.
- Themes: groups of cluster labels that build an emotion's direction.
  - `themes.py`, `scripts/themes.py`, `scripts/scene_classes.py`.
- End-to-end pipeline with checkpoints: harvest, modes, judge, report.
  - `pipeline.py`, `scripts/pipeline.py`, `notebooks/modes_pipeline.ipynb`, `configs/harvest_*.yaml`.

### Reports and views

- Modes report: one card per neuron, both poles, judge verdicts, logit lens.
  - `modes_report.py`, `templates/modes_report.html`, `scripts/modes_report.py`.
- Logit lens of residual directions: `lens.py`.
- Per-token cosine strips with circuitsvis: `token_page.py`, `scripts/token_report.py`, `scripts/token_scores.py`.
- A direction's top neurons at its read position, with prompts at their extremes.
  - `direction_page.py`, `scripts/direction_prompts.py`.

### Interventions

- Remove a theme's neurons from an emotion's write, and score held-out stories.
  - `theme_removal.py`, `scripts/theme_removal.py`, `configs/remove_*.yaml`.

### Experiments only at their tags

These have their code and log at the tag, not on `archive`.

- Emotion vectors at all 36 layers: `exp/journey_all`.
- Sparse sets at every layer: `exp/sparse_layers`.
- Whole-network circuits (failed): `exp/circuits_l25-round1`, `exp/circuits_l25-round2`.
- Local circuits and chains of single neurons: `exp/neuron_circuits`, `exp/chain`, `exp/chain_scans`, `exp/chain_context`.
- Attention heads into MLP neurons: `exp/head_reads`.
- Causal tests of sets: `exp/causal_sets`, `exp/swap_values`, `exp/tail_causal`.
- Gate and up rows read through the embeddings: `exp/input_rows`.
- Negated emotion words: `exp/word_tests`.
- Signal and noise neurons: `exp/signal_noise`, `exp/residual_sets`, `exp/judge_sets`.
- 6-fold accuracy of the emotion vectors: `exp/crossval`.
