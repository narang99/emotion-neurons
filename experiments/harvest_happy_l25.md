# Modes of the happy neurons, both poles, judged blind

Date: 2026-11-21. Branch: `stack/harvest-happy`. Tag: `exp/harvest_happy_l25`. Status: kept.

Config: `configs/harvest_happy_l25.yaml`. Results: `results/harvest_happy_l25/` (report: `report.html`).
Large data: `s3://narang99-private/emotion-neurons/harvest_happy_l25/runs` (corpus pass, 2.3 GB; hits; sources).

```
python scripts/harvest.py      --config configs/harvest_happy_l25.yaml   # L4, 683 s
python scripts/modes.py        --config configs/harvest_happy_l25.yaml   # L4, 1485 s
python scripts/judge_modes.py  --config configs/harvest_happy_l25.yaml   # OpenRouter, ~15 min
python scripts/modes_report.py --config configs/harvest_happy_l25.yaml   # CPU, ~20 s
```

## Question

- Which happy neurons push toward happy while they fire on something else?
- Needs both poles: a neuron that happy turns down serves happy through its low pole.

## Method

- Neurons: the happy greedy set up to 75% of the write (532), and 20 random controls.
- Harvest: 2,000,376 tokens (1,962,696 valid) of fineweb-edu (700k), TinyStories (700k) and UltraChat (600k), to layer 25.
  - Band edges per neuron: p97, p99, p99.9, p99.99 and p3, p1, p0.1, p0.01. Examples per band; 1000 hits per pole.
- Modes: the MLP input at 500 hits per neuron and pole, re-run from the stored windows; PCA and HDBSCAN.
  - Check: the recomputed activation matches the harvest within 0.009 (relative) over 200 hits.
  - Median 3 clusters per neuron and pole; 4483 clusters.
- Judges (OpenRouter, blind, scene not word): Haiku 5.5 and Gemini 2.5 Flash-Lite.
  - Per cluster: a mode name, how many of its 5 windows sit in a happy scene, the scenes.
- Side: the pole happy moves a neuron toward (the sign of its happy shift). Class from the clusters on that side:
  - match: a cluster is a happy scene (3+ of 5) for every judge that judged it;
  - split: for only one judge;
  - no match: for no judge.

## Results

- Judges: Haiku 4483 of 4483 clusters, Gemini 4428 (it skips the same 55 every time). They agree on happy vs not for 83%.
  - Happy scenes: Haiku 22% of clusters, Gemini 24%.

| class | neurons | % of happy push | in top 100 |
|---|---|---|---|
| match | 244 | 44.8% | 57 |
| split | 152 | 11.7% | 21 |
| no match | 136 | 11.9% | 22 |

- Baseline: 46% of the happy-set neurons are "match", against 20% of the 20 controls (TinyStories is full of happy scenes).
- The low pole matters: 10272 (#2) fires most negatively on "They [[hugged]] again" (-40.75).
  - The earlier harvest (high pole only) showed it as love-poem punctuation.
- 3832 (#5) is a laugh-word neuron, at its low pole:
  - On the laugh word itself (772 tokens): mean -6.30, at the low pole 98.3% of the time (base rate 2.9%).
  - Its high pole fires in joyful moments (judges 4/5 and 5/5) and writes "humor, joke, 幽默".
  - Happy stories mention laughing in 83% of cases, so its happy mean goes down (-0.33 vs -0.04).
  - So the side rule (sign of the mean) puts it at "no match", by its low pole.
- 9964 (#7) separates neutral from emotional stories: neutral +1.62; the emotions +0.66 to +1.27, with happy (+0.90) in the middle.
  - Its happy shift comes from the neutral stories in the all-story baseline.
- Flipping 3832 in the happy write (held-out stories): separation 96.60% to 96.63-96.69%, i.e. 1-4 of 4400 pairs, inside the
  range of 9 other flips (-0.52 to +0.19 points). The lens turns to "laughter, laughing, joke, humor".

## Limits

- The side comes from a story-mean shift. A few strong firings at the other pole can hide behind it (3832).
- Held-out stories share the dataset's biases, so held-out separation cannot show a bias.
- 20 controls is a small baseline.
- The run logs were lost with the VM; their lines are quoted above.
