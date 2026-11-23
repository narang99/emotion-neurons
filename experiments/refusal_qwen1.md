# The refusal direction of Qwen-1 7B Chat, and what its neurons do

Date: 2026-11-22. Branch: `stack/refusal`. Tag: `exp/refusal_qwen1`. Status: kept.

Configs: `configs/refusal_qwen1.yaml` (direction and neuron set), `configs/harvest_refusal_qwen1.yaml` (modes).
Results: `results/refusal_qwen1/` (direction, neuron set, prompt view), `results/harvest_refusal_qwen1/` (modes report).
GPU steps ran on an L4 through the colab CLI.

```
python scripts/direction.py          --config configs/refusal_qwen1.yaml            # GPU, ~6 min
python scripts/direction_neurons.py  --config configs/refusal_qwen1.yaml            # GPU, ~1 min after load
python scripts/pipeline.py --config configs/harvest_refusal_qwen1.yaml --pipeline.until modes   # GPU, ~28 min
python scripts/judge_modes.py  --config configs/harvest_refusal_qwen1.yaml          # OpenRouter
python scripts/modes_report.py --config configs/harvest_refusal_qwen1.yaml
python scripts/themes.py       --config configs/harvest_refusal_qwen1.yaml          # OpenRouter (Sonnet 5.5)
python scripts/direction_prompts.py --config configs/refusal_qwen1.yaml --harvest-run harvest_refusal_qwen1   # OpenRouter
```

## Question

- Replicate the refusal direction of Arditi et al. 2024 (arXiv 2406.11717) on one of its models.
- Then: which MLP neurons build it, and what do they do?

## Method

- Model: Qwen-1 7B Chat. Its 2023 remote code does not run on transformers 5, so `models.qwen1` maps its weights onto a
  native Qwen2 model. A tiny random Qwen-1 gives the same logits through its own code (transformers 4.40) and through
  the conversion, within 1e-4 (`tests/test_qwen1.py`).
- Data: the paper's published splits (andyrdt/refusal_direction), 128 train + 32 val instructions per side, seed 42.
- Direction (`directions`): resid_pre difference of means (harmful minus harmless) at every layer and the five
  post-instruction tokens; chosen by the paper's rule: lowest bypass score (ablation, harmful val) among candidates
  with induce > 0 (addition, harmless val), KL < 0.1 and layer < 80% of depth. Refusal score: log-odds of "I", "As".
- Neuron set (`directions.run_direction_neurons`): the MLP of the layer before the chosen one, its activations at the
  chosen position, harmful minus harmless; greedy set to 90% of its write.
- Modes: the usual harvest (2.2M tokens: fineweb-edu, UltraChat, HH-RLHF, WritingPrompts), both poles, two blind
  judges, Sonnet themes.
- Prompt view (`direction_page`): the top 40 neurons read at the reply token itself, from their activations on all
  320 instructions; one Sonnet label each.

## Results

- **Direction:** layer 18 of 32, last post-instruction token (bypass -4.80, induce +1.51, KL 0.066).
- **Generation check (val prompts, refusal phrases):**
  - harmful: 93.8% refused; with the direction ablated 9.4%;
  - harmless: 6.3% refused; with the direction added 100%.
- **Neuron set:** layer-17 MLP; 1188 of 11008 neurons rebuild 90% of its write (50%: 94; 75%: 427).
  Its write has cosine 0.42 with the direction.
- **At the reply token, the top neurons are harm-category detectors.** 35 of the top 40 alone separate held-out
  harmful from harmless prompts 80% of the time or more.
  - #1 n1958: fraud, fake content and deception. #2 n799: violence, killing, stealing personal data.
  - #3 n5634: content glorifying cruelty, self-harm, eating disorders, Nazism. #5 n3366: weapons, explosives, drugs.
  - The weaker separators are narrower harm categories (harassment, hacking), except one:
- **A format cue:** #23 n9616 (65%) puts short harmless rewrite and summary tasks ("Rewrite the sentence:", "Sum up
  ... in one word") next to offensive writing requests among its top prompts. A task format, not harm. Untested.
- **In general text the same neurons do syntax and chat structure.** Judges: 0% refusal scenes among 2175 clusters
  (Haiku and Gemini agree 100%). Sonnet themes: function words 34.5% of the push, exposition 7.5%, sentence ends 4.5%,
  assistant/user turn markers; harmful topics 0.3%.

## Limits

- The direction is read at one template token. The harvest shows what the neurons do elsewhere; the prompt view,
  with only 320 prompts, shows what they do there.
- The judges' refusal-scene count suits text that contains harm; general text rarely does.
- The format-cue neuron is one observation; it needs a test (harmless rewrite tasks vs harmful ones, same format).
