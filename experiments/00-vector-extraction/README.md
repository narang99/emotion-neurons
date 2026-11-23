# Vector extraction

Reports of the two kinds of vectors this repo extracts: emotion vectors, and the refusal direction of Arditi et al. 2024.

## What was done

- No vectors were extracted again here. The reports use the saved results of the earlier runs, made and checked by the collaborator.
  - Emotion vectors: tag `exp/stage2_l25` (Qwen3-8B, layer 25).
  - Refusal direction: tag `exp/refusal_qwen1` (Qwen-7B-Chat).
- This experiment adds the report pipeline (`emotion_neurons.reports`, `scripts/report.py`) and shows its output.
- It also adds circuitsvis token views. Their per-token scores need a forward pass, so they ran on a Colab L4.
  - The scores read the saved vectors. They do not change them.
- The configs here are the ones those runs used. Only `results_dir` changed, to point at this folder.

## Scripts that were run

```
python scripts/token_scores.py direction --config experiments/00-vector-extraction/refusal_qwen1.yaml         # GPU, L4, 84 s
python scripts/token_scores.py emotion_vectors --config experiments/00-vector-extraction/stage2_l25.yaml     # GPU, L4, 74 s
python scripts/report.py emotion_vectors --config experiments/00-vector-extraction/stage2_l25.yaml
python scripts/report.py direction --config experiments/00-vector-extraction/refusal_qwen1.yaml
```

- The scripts that made the saved results (not run again):
  - `scripts/extract.py` and `scripts/build_vectors.py` with `stage2_l25.yaml`;
  - `scripts/direction.py` with `refusal_qwen1.yaml`.

## Results

- `results/stage2_l25/report.html`: emotion vectors.
  - Held-out 12-way nearest-vector accuracy, denoised: 83.8% (`resid_mid`), 82.9% (`resid_post`, `mlp_act`), 80.8% (`mlp_out`). Chance is 8.3%.
  - Split-half cosine (topics 0-9 against 10-19): 0.79-0.92 at `resid_mid`.
  - Also: confusion and emotion x emotion cosine per site, and 192 stories mislabelled at every site.
- `results/refusal_qwen1/report.html`: refusal direction.
  - Chosen: layer 18, last post-instruction token. Bypass -4.80, induce 1.51, KL 0.066.
  - Refusal rate: harmful 93.8% plain, 9.4% ablated; harmless 6.2% plain, 100% added.
  - Also: the selection grid (bypass, induce, KL), norm and cosine of every candidate, and every reply.
- Token views (`token_scores.{json,safetensors}`, fp16):
  - Emotions: all 280 held-out stories, 59305 tokens, cosine with each of the 12 denoised `resid_mid` vectors.
    - Check: the argmax of each story's mean token cosine over tokens 50+ labels 83.8% of the 240 emotional stories.
      This matches the saved held-out accuracy (83.8%).
    - The report shows 2 stories per emotion and 2 neutral ones.
  - Refusal: all 64 val prompts with their plain replies, 5932 tokens, projection onto the unit direction at the input of layer 18.
    - Check: at the last prompt token, harmful 46.3 (sd 14.1), harmless 2.0 (sd 5.9), a gap of 44.4.
      The direction's norm (the same gap on the train prompts) is 56.7.
    - The report shows 6 prompts per category.
  - The circuitsvis views load their renderer from unpkg, so they need a network connection.
