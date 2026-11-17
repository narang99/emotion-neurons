# Emotion neurons in Qwen3-8B: plan

Goal: find the MLP neurons in one late layer of Qwen3-8B that read or write emotion
directions, then look at what those neurons fire on (Goodfire-style, p97 and up).

Sources:
- Anthropic, *Emotion concepts and their function in an LLM* (vector recipe)
- Goodfire, *Neurons as an interpretable basis* (neuron definition, percentile bands, controls)
- narang99, *Emotion vectors in Hindi* (cheap recipe on Gemma-2-9B, L4, what worked)

## Fixed choices

- **Model:** `Qwen/Qwen3-8B`, bf16, Colab **L4** (24 GB GPU; standard RAM is enough). No quantization, so neuron statistics stay clean.
- **Layer:** **L = 25** of 36 (about 70% depth). One layer only; another layer is a new experiment that reruns the same pipeline.
- **Sites** (a site is a point in the model where activations are recorded). Qwen3 decoder layer: `h += attn(ln1(h)); h += mlp(ln2(h))`.
  - `resid_mid`: input to `layers[25].post_attention_layernorm`, the residual stream after attention and before the MLP (4096).
  - `resid_post`: output of `layers[25]`, the residual stream after the MLP (4096).
  - `mlp_act`: input to `layers[25].mlp.down_proj`, `a = silu(gate) * up` (12288 neurons).
  - `mlp_out = resid_post − resid_mid` is derived, not hooked.
- **Neuron = `a_n`**, bipolar. The positive and negative poles are analysed separately (Goodfire).
- **Experiments are YAML configs** read by a pydantic-settings model (the in-tfm `run_config.py` pattern).
  - Everything that distinguishes one run from another lives in `configs/*.yaml`, so each commit reproduces a run.
  - Unknown keys fail (`extra="forbid"`); command-line flags override the file for one-off changes.

## Stage 1: stories (done)

- 12 emotions × 20 topics × 6 stories = **1440 emotional stories**, plus **240 neutral** (20 topics × 12), in `data/stories/<emotion>.jsonl`.
  - Emotions: happy, inspired, loving, proud, calm, desperate, angry, guilty, sad, afraid, nervous, surprised.
- Written by Claude subagents (never name the emotion; mix first and third person; 120–220 words).
- `scripts/validate_stories.py` (format, counts, length, banned words) and `scripts/audit_stories.py` (labels match file, duplicates, leakage) pass.
- Spot-read by the user, 2 stories per emotion: good enough.

## Stage 2: activations, vectors and a check that the vectors work (current)

### 2a. Extract (`scripts/extract.py`, GPU, one pass over all 1680 stories)

- Plain story text, no chat template; batched with right padding, and the padding is masked out of every sum.
- Hooks at layer 25 only: `resid_mid`, `resid_post`, `mlp_act`.
- **Stored per story:** the sum of each site over tokens 0–49 and over tokens 50 to the end, plus both token counts.
  - Each value is cast to fp32 before adding. fp32 cannot overflow here (max ~3e38); summing in bf16 would lose precision.
  - Any average can be rebuilt later without the model: from token 50 (paper), all tokens, or the first 50.
  - Size: 1680 × (4096 + 4096 + 12288) × 2 ranges × 4 B ≈ **275 MB**, saved as safetensors on Drive.
- No per-token activations are stored. A later experiment reruns the pass.

### 2b. Vectors (`scripts/build_vectors.py`, CPU, from the stored sums)

- **Split:** story_idx 5 of every emotion × topic is held out (20 per emotion); for neutral, the last 2 per topic (40).
  Vectors are built from the remaining 100 per emotion and 200 neutral.
- **Average:** from token 50 onwards (paper).
- Per site × emotion:
  - baseline: minus the mean over the 12 emotions, and minus neutral;
  - raw and denoised (project out the neutral PCs that explain 50% of variance, fitted per site; the PCs are saved).
- Output: `vectors.safetensors` = 12 emotions × (4096 + 4096 + 4096 + 12288) × 2 baselines × 2 variants × 2 B ≈ **2.4 MB**, plus `vectors_meta.json`. Both committed to git.

### 2c. Does each vector work? (`scripts/project.py`, GPU, held-out stories only, about 280 stories)

- A short second pass over the held-out stories. Each token's activation is projected onto every unit emotion vector, for every site.
- Saved per story: tokens plus a tokens × 12 emotions × sites projection array (small). Kept in git for later review.
- **Report** (static HTML, `reports/<run_id>/`):
  - circuitsvis `colored_tokens_multi` per held-out story: the token text coloured by each emotion's projection, with a selector for the emotion.
  - A 13 × 12 matrix: mean held-out projection (tokens 50+) for each story emotion (rows, plus neutral) against each vector (columns). The vectors work if the diagonal dominates.
  - Held-out accuracy: each story's nearest vector, using the story's mean activation.
  - Split-half cosine per emotion: is the vector stable with 100 stories?
  - Emotion × emotion cosine heatmap: is there valence structure?
  - Logit lens on the `resid_post` vectors: top tokens per emotion.
- **Story check:** the nearest-vector test runs on every story (leave-one-out over the train set as well). One Haiku agent reads only the flagged stories and says whether each is mislabelled or a near-neighbour confusion (e.g. nervous vs afraid).

## Stage 3: the neurons behind the MLP's write (`scripts/neuron_set.py`, CPU only)

Everything stays inside layer 25's MLP. Only the neuron `a_n = silu(gate_n) · up_n` (Goodfire) is treated as a
privileged basis; the residual dimensions are not, which is why a large share of them is kept.

- **The write:** `δ_e = mean(resid_post − resid_mid)` for emotion e minus the mean over emotions (= `mlp_out/mean/raw` from Stage 2).
  Exactly, `δ_e[d] = Σ_n W_down[d, n] · Δā_{e,n}`, where `Δā_e` = mean `mlp_act` for e minus the mean over emotions.
- **Dimensions:** the dimensions of `δ_e` covering 90% of its positive mass, and separately 90% of its negative mass.
- **Neurons per dimension:** for each kept d, the terms `W_down[d, n] · Δā_{e,n}`; keep the neurons covering 90% of the
  positive terms and, separately, 90% of the negative terms. Both poles are kept, so the modes of neurons that pushed
  the dimension up can be compared with those that pushed it down.
- **The set:** the union over dimensions, per emotion and pole, then over emotions. Each neuron is tagged with
  (emotion, dimension, pole, share).
- **Saved:** the full per-dimension contribution scores, so other thresholds (on dimensions or neurons) can be rebuilt without the model.
- **Printed before anything else runs:** the set size per emotion and pole, and in total.
- `W_down` of layer 25 comes from the model's safetensors shard (no GPU).

## Stage 4: high activations for the chosen neurons (`scripts/harvest.py`, GPU)

- Neurons: the Stage 3 set plus 50 random control neurons.
- Corpus: about 2M tokens of FineWeb-edu, TinyStories and UltraChat, in fixed-length windows tagged by document id.
- Run the model only up to layer 25. Write `a_n` for the chosen neurons to a memmap (2M × n_neurons × fp16).
- Per neuron: p97, p99, p99.9, p99.99 of `a_n`; up to 50 examples per band, 60 tokens before the firing token and 8 after.
- Output: per-neuron JSON plus a static HTML report; copied to S3.

## Stage 4b: modes (HDBSCAN first, Haiku names them)

- **Cluster before reading.** Per neuron, the p97+ hits are clustered with HDBSCAN on the neuron's input at the hit
  token (the normalised `resid_mid`, cosine via L2-normalise + euclidean); sklearn's HDBSCAN is enough for ≤ 1k hits.
  - Many hits will be noise; that's fine, the clusters still show the modes.
  - Hits outside a cluster (noise, or hits beyond the clustered sample) are assigned to the nearest cluster centroid
    by cosine, above a cutoff; below it they stay "other".
- **Haiku reads only exemplars:** about 5 snippets per cluster plus a few noise hits, and names each mode.
  This replaces reading 50 snippets per neuron. The set size and token estimate are printed first.
- Default cap if the set is large: top 10 neurons per emotion per pole + 20 random controls, ≤ ~1.5M Haiku tokens.
- Compare the modes of neurons that pushed a dimension up with those that pushed it down, per emotion.

## Stage 5: analysis (decided after looking)

- Is a writer neuron's p97+ text about the emotion, or about something correlated with it?
- Do writers and readers overlap? How many neurons rebuild δ? Do negative emotions share neurons?
- Ablate the top writers and measure the drop in the vector's projection on held-out stories.

## Repo layout (uv)

```
emotion_neurons/   config.py (pydantic-settings), model.py (load + hooks), stories.py, vectors.py, project.py, report.py
scripts/           extract.py, build_vectors.py, project.py (later: attribute.py, harvest.py)
configs/           emotions.json, topics.json, <experiment>.yaml
```

- `pyproject.toml` floors stay at or below Colab's preinstalls (in-tfm pattern), so torch and numpy are never reinstalled.
- Colab: clone → `uv pip install --system --no-deps -e .` → only the deps Colab lacks.
- Large outputs (pooled sums) go to Google Drive under `emotion-neurons/<run_id>/`. Vectors, projections and reports are committed.

## Rough cost on an L4

| stage | time |
|---|---|
| Extract, 1680 stories | ~10 min |
| Vectors | seconds |
| Project + report, ~280 stories | ~2 min |
