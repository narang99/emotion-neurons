# Stage 3–4: the layer-25 neurons behind the MLP's emotion write

Run 2026-11-18 on a Colab L4, commit `59949ed` (branch `stage2`), config `configs/stage3_l25.yaml`.
Outputs: `results/stage3_l25/` (committed) and `s3://narang99-private/emotion-neurons/stage3_l25/` (also the
2,000,432 × 469 activation memmap and token arrays).

```
python scripts/neuron_set.py --config configs/stage3_l25.yaml   # CPU, seconds
python scripts/harvest.py    --config configs/stage3_l25.yaml   # 10 min on an L4
python scripts/modes.py      --config configs/stage3_l25.yaml   # 12 min
# Haiku names the modes: results/stage3_l25/modes/batch_XX.txt -> names_XX.jsonl (5 agents, ~760k tokens)
python scripts/summarize_modes.py --config configs/stage3_l25.yaml
```

## Log

- **Round 1: does the 90% / 90% rule pick a set?** No: it picks every neuron.
  - The write is dense. Per emotion, 1264–1376 dims per sign cover 90% of `δ_e`; each dim needs a median of ~2970 neurons for 90% of its positive terms.
  - The union is 11214–11318 neurons per emotion and all 12288 over emotions. Even 50% of dims × 10% of terms gives 1339–2148 neurons per emotion (sweep in `neuron_set.json`).
  - `W_down @ Δā` rebuilds `δ_e` to 0.07–0.12% (check that the write is the MLP's).
- **Round 2: rank inside the set.** Score per neuron = Σ over kept dims of `sign(δ_d) · W_down[d, n] · Δā_n`; pos = highest, neg = lowest.
  - First version clamped each dim's term before summing: 277 of 300 pos/neg slots were the same neuron within an emotion (big neurons won both poles). Fixed to a net sum; pos and neg are disjoint (asserted). The first harvest was discarded.
  - Top 25 per (emotion, pole): 419 unique neurons + 50 random controls harvested; 320 of the 419 are in one group only.
- **Round 3: where do they fire?** Positive-pole neurons fire on simple narrative.
  - Share of p97+ hits from TinyStories: pos 0.52, neg 0.35, controls 0.26 (corpus is 0.32 TinyStories by tokens).
- **Round 4: what are the modes?** Top 10 per (emotion, pole) + 20 controls = 196 neurons; HDBSCAN on the MLP input at ≤ 562 p97+ hits each.
  - Median 3 clusters per neuron (max 11); median 12% of hits left unclustered. All 997 clusters named by Haiku.
  - Modes whose name mentions a feeling word (regex over Haiku's names: feel, felt, scared, proud, sad, ...):

    | group | neurons | modes | share of modes | share of hits |
    |---|---|---|---|---|
    | pos | 70 | 341 | 0.30 | 0.47 |
    | neg | 109 | 412 | 0.20 | 0.28 |
    | controls | 20 | 68 | 0.15 | 0.21 |

  - pos > neg for 10 of 12 emotions (exceptions: desperate 0.27 vs 0.30, guilty 0.21 vs 0.33).
  - Haiku's strict own-emotion label is rare: 10 of 523 pos (group, mode) pairs, 4 of 459 neg, 1 of 68 controls (any emotion).
- **Examples (share of the neuron's p97+ hits):**
  - 3068 (proud/pos): "proud" or "and" right after "very"/"so" (very proud), 98%. A clean proud neuron.
  - 11109 (sad/pos, afraid/pos): "scared" or "and" right after nervous/scared words, 98%.
  - 48 (guilty/pos): "and" right after an emotion adjective (ashamed, stunned, embarrassed), 94%. Fires after any feeling word, not guilt alone.
  - 2066 (angry/pos): "and", "she", "happy" after a feeling or smiling verb, 90%.
  - 3634 (guilty/pos): the next word after sad or hurt events, 90%.
  - 10272 (happy/pos, proud/pos): "boy"/"girl" after "the little" in children's stories, 96%. Story structure, not emotion.
  - 3582 (angry/pos): the next word after characters become glad or proud, 94%. A positive-emotion mode on the angry pole (see below).

## Surprises

- **The top writers do not split by valence.** Pos-pos overlap between two emotions is 0.20 for the same valence and 0.19 for opposite valence; a pos neuron of a positive emotion is almost never a neg neuron of a negative one (0.01).
  - Several top writers fire after *any* feeling word (48, 2066, 3582). They are "a feeling was just named" neurons; the emotion-specific part of the write likely comes from the many small terms, not the top 10.
  - Not tested yet: ranking by the emotion-specific part (e.g. `δ_e` minus the mean write over emotions) instead of the raw write.
- **Duplicated corpus text.** 3–7% of modes are near-identical copies of one passage (mostly UltraChat). Kept; flagged "DUPLICATE:" in `names_XX.jsonl`.
- **The keyword measure is ours, not Haiku's.** The emotion-keyword shares come from a regex over Haiku's mode names; only the counts above were tested.
