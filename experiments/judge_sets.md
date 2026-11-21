# Blind judging: neurons only in the signal set vs neurons in both sets

Date: 2026-11-21. Branch: `stack/06-raw-sets`. Tag: `exp/judge_sets`. Status: kept.

Config: `configs/judge_sets.yaml`. Prompt: `data/judge_prompt.md`. Results: `results/judge_sets/` (selection = key, snippets, judgments, summary).

```
python scripts/judge_select.py --config configs/judge_sets.yaml --device cpu   # CPU
python scripts/judge_snippets.py --config configs/judge_sets.yaml                # L4, 6 minutes
# 4 Haiku judges, 45 neurons each, each in its own folder
python scripts/judge_aggregate.py --config configs/judge_sets.yaml --device cpu
```

## Question

- Method 2 (signal set, about 1900 neurons) adds about 1100 neurons per emotion that method 1 (match set) does not have.
- Do these extra neurons look emotion-related, as often as the match-set neurons?

## Method

- 4 emotions: happy, sad, afraid, proud. The match set is 91% to 92% inside the signal set (happy, proud), 78% to 82% (sad, afraid).
- Per emotion: 20 neurons only in the signal set, 20 in both sets, 5 random neurons in neither. 180 neurons.
- Snippets per neuron: 12 top and 5 bottom contexts on general text; 8 top contexts on all train stories (labels hidden).
- Judges: 4 Haiku agents, blind to the group. Per neuron: theme, related to the named emotion (0 / 1 / 2), related to any emotion, kind.

## Results

| group | n | related to the named emotion (>= 1) | clearly (= 2) | any emotion | syntax | unclear |
|---|---|---|---|---|---|---|
| in both sets | 80 | 0.21 [0.14, 0.31] | 0.03 | 0.26 | 0.39 | 0.20 |
| only in the signal set | 80 | 0.21 [0.14, 0.31] | 0.00 | 0.29 | 0.36 | 0.29 |
| random | 20 | 0.00 [0, 0.16] | 0.00 | 0.10 | 0.30 | 0.45 |

- **The extra neurons look as emotion-related as the match-set neurons.** Both groups: 21%. Random: 0%.
- **Most neurons in both groups do not look emotional from their contexts.** About 37% are syntax neurons; 20% to 29% are unclear.
- **The related ones are mostly situations and body signs, not emotion words.**
  - Only in the signal set: "physical strain: burning chest, ragged breath" (sad), "leaving quickly: getaway, drive off" (afraid), "eerie whispers and sounds" (afraid), "telling or sharing news" (happy).
  - In both: "fleeing, exile and escape" (sad), "eyes, throat or voice reacting" (afraid), "pride, chest swelled" (proud, n3068).
- Per emotion, related share (both / only signal): afraid 0.25 / 0.30, sad 0.30 / 0.35, happy 0.15 / 0.20, proud 0.15 / 0.00.

## Conclusion

- By how they look, the neurons method 2 adds are no different from the match-set neurons.
- In both sets, about 1 neuron in 5 looks related to the emotion. The rest look like syntax or nothing in particular.

## Limits

- One judge per neuron; no agreement check between judges.
- Contexts show where a neuron fires most. A neuron can matter for the emotion through moderate firing that its top contexts do not show.
