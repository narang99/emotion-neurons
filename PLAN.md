# Emotion neurons in Qwen3-8B: plan

Goal: find the MLP neurons in one late layer of Qwen3-8B that read or write emotion
directions, then look at what those neurons fire on (Goodfire-style, p97 and up).

Sources:
- Anthropic, *Emotion concepts and their function in an LLM* (vector recipe)
- Goodfire, *Neurons as an interpretable basis* (neuron definition, percentile bands, controls)
- narang99, *Emotion vectors in Hindi* (cheap recipe on Gemma-2-9B, L4, what worked)

## Fixed choices

- **Model:** `Qwen/Qwen3-8B`, bf16, Colab **L4** (24 GB). No quantization, so neuron statistics stay clean.
- **Layer:** **L = 25** of 36 (about 70% depth). A single MLP to begin with.
- **Sites (Qwen3 decoder layer: `h += attn(ln1(h)); h += mlp(ln2(h))`):**
  - `resid_mid` = input to `layers[25].post_attention_layernorm` (before the MLP)
  - `resid_post` = output of `layers[25]` (after the MLP)
  - `mlp_act` = input to `layers[25].mlp.down_proj`, 12288 neurons, `a = silu(gate) * up`
  - Also `gate_pre`, `up_pre`, used only for the Q1–Q4 sign tags
- **Neuron = `a_n`**, bipolar. The positive and negative poles are analysed separately (Goodfire).

## Stage 1: stories (Claude subagents), committed to git, done once

- **Emotions (12, the paper's focus set, covering most of its 10 clusters):**
  happy, inspired, loving, proud, calm, desperate, angry, guilty, sad, afraid, nervous, surprised.
- **Topics:** 20 everyday premises (paper style), stored in `configs/topics.json`.
- **Counts:** 12 emotions × 20 topics × 6 stories = **1440 emotional stories**, plus **240 neutral** (20 topics × 12).
  - The paper uses about 1200 per emotion. We use 120.
- **Generator:** one Claude subagent per emotion, plus two for neutral, using the paper's prompt rules
  (never name the emotion; mix first and third person; 120–220 words).
  - As in the paper, the stories come from a different model than the one we read activations from.
  - No GPU time is needed.
- **Validation:** `scripts/validate_stories.py` checks:
  - JSON and ids;
  - per-topic counts;
  - length;
  - banned words from `configs/emotions.json` (for neutral stories, the union of every list).
- **Output:** `data/stories/<emotion>.jsonl`, one record per story: `{id, emotion, topic_id, story_idx, text}`. Story text only.

## Stage 2: activations and vectors (`scripts/extract.py`), kept forever

Rule: a GPU pass is costly, so **store everything it produced and derive later**.
Nothing is thrown away after it is reduced.

- **Single pass:** feed the plain story text (no chat template), batched with right padding.
  - Hooks at **every layer (0–35)**: `resid_mid` (before the MLP), `resid_post` (after the MLP),
    `attn_out` and `mlp_out`.
  - Hook at layer 25 only: `mlp_act` (12288), plus `gate_pre` and `up_pre`.
- **Pooled sums per story, two token ranges:** the sum over tokens 0–49 and the sum over tokens 50 to the end, plus the token counts.
  - Any of these means can be rebuilt later without the model: from token 50 (paper), all tokens, or the first 50.
  - fp32, so the sums don't lose precision.
- **Per-token at layer 25:** per-token `resid_mid`, `resid_post` and `mlp_act` for every story, in fp16.
  - About 1680 × 250 × 20k × 2 B ≈ 17 GB, stored as sharded safetensors on Drive.
  - These are needed for:
    - per-token projections on the stories themselves;
    - which neurons fire on which story tokens;
    - re-pooling any other way.
- **Vectors derived (`scripts/build_vectors.py`, CPU, from the stored sums):**
  - Every site × every layer × every emotion.
  - Both baselines: minus the mean over emotions, and minus neutral.
  - Raw and denoised (neutral PCs at 50% variance, fitted per site and layer); the PCs are saved too.
  - The direction the MLP wrote, `v_post − v_mid`, is one entry among these, not the only one kept.
- **Outputs:**
  - Pooled sums and per-token shards go to Drive.
  - `vectors.safetensors` (all sites, layers, variants: about 12 × 36 × 4 × 4096 × 4 variants × 2 B ≈ 56 MB) plus `vectors_meta.json` are committed to git.
- **Checks before going further:**
  - Split-half cosine per emotion: is the vector stable with 120 stories?
  - Emotion × emotion cosine heatmap: does it show valence structure?
  - Logit lens on `resid_post` vectors: top tokens per emotion.

## Stage 3: from direction to neurons (`scripts/attribute.py`), CPU only, no model forward

- **Write direction:** `δ_e = v_post − v_mid`.
  - Because pooling is linear, raw `δ_e` is exactly the emotion vector of the MLP output: `δ_e = W_down · Δā_e`,
    where `Δā_e` = mean `mlp_act` for emotion e minus the mean over emotions.
- **Writers (who produced δ):** per-neuron contribution `c_n = Δā_{e,n} · (W_down[:,n] · δ̂_e)`.
  - Summed over all neurons this gives exactly `‖δ_e‖`.
  - Rank neurons by `c_n` and report how many neurons make up 50% and 80% of it.
- **Per-dimension view (as asked):** `C[d, n] = W_down[d,n] · Δā_{e,n}`, and the top neurons for each residual dimension d.
  - This is kept as secondary. The residual basis isn't privileged, apart from Qwen's massive-activation dims, which we flag.
- **Readers (who the incoming emotion state drives):** fold RMSNorm into the read vectors (Goodfire),
  `v_gate,n = w_gate,n ⊙ γ / rms̄`, with rms̄ = the mean rms at `resid_mid` on stories.
  - Score `v_gate,n · v_mid_e` and `v_up,n · v_mid_e`, which gives the predicted pre-activation shift.
- **Sets per emotion:** top-k writers (k=32), top-k readers, and their overlap.
  - Also overlap across emotions: do angry and afraid share neurons?
- **Controls (Goodfire):**
  - the same scores for random unit directions with matched norm;
  - a random-neuron set, carried through stage 4.

## Stage 4: high activations for the chosen neurons (`scripts/harvest.py`)

- **Neurons:** the union of the stage-3 sets (≤ about 600) plus 50 random control neurons.
- **Corpus:** about 2M tokens, in fixed-length windows tagged by document id:
  - FineWeb-edu (general)
  - TinyStories (narrative; the blog found Pile didn't work and TinyStories did)
  - UltraChat (chat)
- **Speed:** run the model only up to layer 25, about 30% less compute. Write `a_n` for the chosen neurons, plus the gate/up sign bits,
  to a memmap: 2M × 650 × fp16 ≈ 2.6 GB.
- **Percentiles per neuron:** p3, p1, p0.1, p0.01 (negative pole) and p97, p99, p99.9, p99.99 (positive pole).
- **Samples:** up to 50 examples per band per pole: p97–99, 99–99.9, 99.9–99.99, 99.99+, and the top-50.
  - Window: 60 tokens before the firing token and 8 after (Goodfire).
  - Each example is tagged with its quadrant Q1–Q4. Gate-closed (Q3/Q4) firings are shown separately.
- **Output:** a per-neuron JSON plus a static HTML report (token heatmaps).
  - The emotion-story activations of the same neuron are shown alongside, for comparison.

## Stage 5: analysis (decided after looking)

Candidates:
- Is a writer neuron's p97+ text about the emotion, or about something correlated with it?
- Do writers and readers for the same emotion overlap?
- How many neurons are needed to rebuild δ?
- Are negative emotions served by shared neurons?
- Ablate the top writers and measure the drop in the vector's projection on held-out stories.

## Repo layout (uv)

```
emotion_neurons/   model.py (load + hooks), stories.py, vectors.py, attribution.py, harvest.py, report.py
scripts/           gen_stories.py, extract.py, attribute.py, harvest.py
configs/           emotions.yaml, topics.yaml, run config
```

- Colab: clone → `uv pip install --system --no-deps -e .` → light deps.
  - torch and numpy are not pinned above Colab's versions.
- Large outputs go to Google Drive under `emotion-neurons/<run_id>/`. Small vectors and summaries are committed.

## Rough cost on an L4

| stage | time |
|---|---|
| Stories | ~30 min |
| Extract | ~10 min (batched) |
| Attribute | seconds |
| Harvest, 2M tokens to layer 25 | ~15–25 min |
| Reports | minutes |
