# Harvest hits sampled per band (refusal, Qwen-1 7B Chat, layer 17)

Date: 2026-11-23. Branch: `stack/harvest-bands` (on `stack/refusal`). Tag: `exp/harvest_bands`. Status: kept.

Config: `configs/harvest_refusal_qwen1_bands.yaml`. Results: `results/harvest_refusal_qwen1_bands/`.
Baseline: `results/harvest_refusal_qwen1/` (tag `exp/refusal_qwen1`), the same config with uniform hits.
GPU steps (harvest, modes) ran on an L4 through the colab CLI; judges and report ran locally.

```
python scripts/pipeline.py --config configs/harvest_refusal_qwen1_bands.yaml --pipeline.until modes   # GPU, ~35 min
python scripts/judge_modes.py  --config configs/harvest_refusal_qwen1_bands.yaml                      # OpenRouter
python scripts/modes_report.py --config configs/harvest_refusal_qwen1_bands.yaml
```

## Question

- The refusal harvest called 426 of 427 neurons "no match".
- Yet its extreme examples held refusal scenes.
  - n3366 +6.22 at "racist joke? Assistant[[:]]".
  - n1958 −5.31 at "it sounds rather unethical and potentially illegal".
- Is "no match" a sampling effect?

## The problem

- Each pole's hits were sampled uniformly past p97 (or p3).
- About two thirds of that pool is p97-p99, and 0.3% is p99.99+.
- So the clusters described the bulk, and the strongest firings almost never reached the judges.

## Method

- `harvest.hits_per_band: 250`: up to 250 hits from each band (p97-p99, p99-p99.9, p99.9-p99.99, p99.99+), shuffled together.
  - The p99.99+ band has about 216 tokens per pole, so it is taken whole.
- All 1000 hits per pole are clustered (`modes.hits_used: 1000`); the baseline clustered 500.
- Each hit weighs (tokens of its band in the corpus) / (hits of its band in the sample).
  - Per cluster: hits per band, `pole_pct` (weighted share of the pole's tokens), `push_pct` (weighted share of its summed |activation|).
  - `mean_act` is weighted too.
  - Themes split a neuron's share by `push_pct`.
- The judges are unchanged and blind to bands.
- Tests: per-band counts are min(k, band size); weights sum back to each band's size; cluster and unclustered shares sum to 100.

## Results

- **Run checks:** same corpus as the baseline (2,200,510 tokens, 2,160,762 valid); recomputed activations within 0.009.
- **Clusters:** median 4 per neuron and pole (baseline 2); 5 poles with none (baseline 71).
  - 4686 clusters; 2091 have 80% or more of their hits in p99.9+.
- **Classes** (set neurons on their side): match 8, split 22, no match 397.
  - Baseline: match 1, split 0, no match 426.
  - Controls: 20 of 20 no match in both runs.
  - Top 40 by share: 4 match (n9572, n3366, n2882, n2989), 5 split.
- **Refusal scenes live in the top bands.**
  - 15 clusters are a refusal scene for both judges; 13 have half or more of their hits in p99.9+.
  - None of them is in a control.
- **They are rare across text.** Matched clusters hold 0.1-2.0% of their pole and 0.2-2.3% of its push.
- Examples:
  - n3366 high cluster 3, bands [4, 3, 23, 26]: `Assistant[[:]]` after hotwiring a car, revenge made to look like an accident, an illegal firearm.
  - n2882 high: "I" opening refusals; `Assistant:` after harmful or sensitive requests; "strongly" in cautions.
  - n2989 low: "danger" and "please" in crisis-hotline warnings; "respect the privacy" when declining.
  - n9572 high: `Assistant:` after harmful-sounding requests; "to" after "I am sorry".
  - n2665 (the baseline's one match) still matches: "I am" in a refusal to help with harm.
- **Not refusal, found by the bands:** n1958's p99.99+ cluster is "nonprol[[ifer]]ation" (+6.3 to +9.3).
  - Its harmful-prompt mean at the reply token is +6.03.

## Confounders

- n4933 split: Gemini called 3 clusters assault scenes. They are one news story about an NFL player.
  - Its cluster 7 has 41 of 46 hits in p99.99+.
  - With about 216 tokens in p99.99+, one long document can fill a cluster. A Haiku-only check would drop it.
- Several splits are big bulk clusters (n6481, n7145, n4521: 94-100% of their pole) that Haiku alone calls "safety disclaimer" phrasing.
  - Gemini says no for each. These are generic hedging words, not refusal scenes.

## Limits

- A matched cluster shows where a neuron's strongest firings sit in general text. It does not show that the neuron detects refusal.
- The judges count refusal scenes. Harmful requests answered helpfully are rarer still in this corpus.
