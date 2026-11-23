# Setup and usage

## Setup

```
git clone git@github.com:narang99/emotion-neurons.git && cd emotion-neurons
uv sync
ln -s ../../scripts/git-hooks/pre-commit .git/hooks/pre-commit
bash scripts/fetch_runs.sh    # runs/stage2_l25: pooled story sums, 263 MB, public
```

- GPU steps run on Colab (an L4 is enough).

## Emotion vectors (Qwen3-8B, layer 25)

```
python scripts/extract.py --config experiments/00-vector-extraction/stage2_l25.yaml                    # GPU, about 10 min
python scripts/build_vectors.py --config experiments/00-vector-extraction/stage2_l25.yaml --device cpu  # CPU, seconds
```

- Data: 1680 stories in `data/stories/`: 12 emotions x 20 topics x 6 stories, and 240 neutral stories.
  - Held out: story_idx 5 of each emotion, neutral story_idx 10 and 11.
- `extract.py` keeps per-story sums of four sites over tokens 0-49 and 50+: `resid_mid`, `resid_post`, `mlp_act`, `mlp_out`.
  - `fetch_runs.sh` gets its output, so `build_vectors.py` needs no GPU.
- A vector is the mean over an emotion's train stories (tokens 50+) minus the mean over the 12 emotions.
  - `denoised`: the top neutral-story PCs (50% of variance) are projected out.
- Output: `experiments/00-vector-extraction/results/stage2_l25/vectors.safetensors`, keys `<site>/<mean|neutral>/<raw|denoised>`, each `[12, dim]`.

## A difference-of-means direction (Arditi et al. 2024)

```
python scripts/direction.py --config experiments/00-vector-extraction/refusal_qwen1.yaml   # GPU, about 6 min
```

- `datasets.InstructionSet` samples instructions per category in the model's chat format.
- The script picks a layer and post-instruction position by the effect of ablating and adding the direction.
  - It then checks the direction by generation, plain, ablated and added.
- Model families are in `models.adapters`: `hf` (native models) and `qwen1` (Qwen-7B-Chat converted to Qwen2).
- Output: `experiments/00-vector-extraction/results/refusal_qwen1/direction.{json,safetensors}`, `replies.json`.

## Neuron poles

```
python scripts/poles.py --config configs/poles.yaml   # GPU, about 10 min
```

- p3 and p97 of each neuron of one layer, over 210k tokens of fineweb-edu, TinyStories and UltraChat.
- Output: `results/poles/thresholds.safetensors`, `p3` and `p97`, `[neuron]` each.

## Reports

```
python scripts/token_scores.py emotion_vectors --config experiments/00-vector-extraction/stage2_l25.yaml   # GPU, about 1 min
python scripts/token_scores.py direction --config experiments/00-vector-extraction/refusal_qwen1.yaml      # GPU, about 1.5 min
python scripts/report.py emotion_vectors --config experiments/00-vector-extraction/stage2_l25.yaml   # CPU, seconds
python scripts/report.py direction --config experiments/00-vector-extraction/refusal_qwen1.yaml      # CPU, seconds
```

- A report reads a run's saved results and writes `report.html` next to them. It loads no model.
- `emotion_neurons.reports`: a builder reads the results into Pydantic data, a Jinja template renders it.
  - A new report is a builder, a template and one entry in `REPORTS`.
- `token_scores.py` scores every token of the held-out stories or val prompts against the saved vectors.
  - The reports show these as circuitsvis token views. Without them, a report has no token views.
- The direction report downloads the val instructions from the config's sources, to show each prompt with its reply.
