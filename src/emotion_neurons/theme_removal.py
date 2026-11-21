"""Remove a theme's neurons from an emotion's MLP write, and compare the two vectors on held-out stories.

- Neurons to remove: those whose clusters on the emotion's side have the theme as their only emotion theme;
  their other clusters may only be non-emotion themes (style, syntax, off-topic). From a modes run's
  cluster_themes.csv (`emotion_neurons.themes`).
- Vectors, from train stories only (held-out stories never touch them):
  - d = W @ a, a[n] = mean activation on the emotion's train stories - mean on all train stories (tokens 50+);
  - d_minus = d - sum over removed neurons of a[n] w_n (each removed neuron held at its all-story mean).
- Held-out score of a story: its mean MLP output minus the train mean output, projected on the unit vector.
- Token score (`token_scores`): cosine of a token's MLP output, minus the train mean output, with each vector.
  - The mean is subtracted because raw outputs are dominated by a few large dimensions shared by every token.
"""

import csv
import json
from pathlib import Path

import torch
from jaxtyping import Float
from safetensors.torch import save_file
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .capture import late_means
from .config import ThemeRemovalRunConfig
from .emotions import EMOTIONS
from .models import load_w_down, read_layer
from .poles import padded_batch
from .texts import StorySplit, load_stories, load_story_records


def only_theme_neurons(cluster_themes: Path, theme: str, not_emotion: set[str]) -> list[int]:
    themes: dict[int, list[str]] = {}
    for row in csv.DictReader(cluster_themes.open()):
        themes.setdefault(int(row["neuron"]), []).append(row["theme"])
    return sorted(n for n, t in themes.items() if theme in t and all(x == theme or x in not_emotion for x in t))


def chance_above(a: Tensor, b: Tensor) -> float:
    """Chance that a random element of a scores above a random element of b (ties count half)."""
    diff = a[:, None] - b[None, :]
    return float((diff > 0).double().mean() + 0.5 * (diff == 0).double().mean())


def build_vectors(cfg: ThemeRemovalRunConfig, removed: list[int]) -> dict[str, Tensor]:
    A = late_means(cfg.runs_dir / cfg.sums_run)["mlp_act"].double()
    split = StorySplit.of(load_story_records(cfg.runs_dir / cfg.sums_run))
    own = split.label == EMOTIONS.index(cfg.emotion)
    abar = A[split.train].mean(0)
    shift = A[own & split.train].mean(0) - abar
    W = load_w_down(cfg.model, cfg.layer).double()
    d = W @ shift
    index = torch.tensor(removed, dtype=torch.long)
    return {"d": d, "d_minus": d - W[:, index] @ shift[index], "mean_out": W @ abar, "shift": shift}


def heldout_scores(cfg: ThemeRemovalRunConfig, vectors: dict[str, Tensor]) -> dict:
    A = late_means(cfg.runs_dir / cfg.sums_run)["mlp_act"].double()
    split = StorySplit.of(load_story_records(cfg.runs_dir / cfg.sums_run))
    W = load_w_down(cfg.model, cfg.layer).double()
    out = (A[split.heldout] - A[split.train].mean(0)) @ W.T
    labels = split.label[split.heldout]
    own = labels == EMOTIONS.index(cfg.emotion)
    result = {}
    for name in ("d", "d_minus"):
        score = out @ (vectors[name] / vectors[name].norm())
        result[name] = {
            "separation": chance_above(score[own], score[~own]),
            "mean_by_label": {
                (EMOTIONS[i] if i >= 0 else "neutral"): float(score[labels == i].mean()) for i in labels.unique().tolist()
            },
        }
    result["cos_d_d_minus"] = float(torch.cosine_similarity(vectors["d"], vectors["d_minus"], 0))
    return result


def run_theme_removal(cfg: ThemeRemovalRunConfig) -> dict:
    """CPU step: results/<run_id>/{vectors.safetensors, removal.json}."""
    r = cfg.removal
    removed = only_theme_neurons(cfg.results_dir / r.modes_run / "cluster_themes.csv", r.theme, set(r.not_emotion_themes))
    vectors = build_vectors(cfg, removed)
    scores = heldout_scores(cfg, vectors)
    shares = {
        int(row["neuron"]): float(row["share"]) for row in csv.DictReader((cfg.results_dir / r.modes_run / "neurons.csv").open())
    }
    summary = {
        "emotion": cfg.emotion,
        "theme": r.theme,
        "removed": removed,
        "removed_push_pct": round(sum(shares[n] for n in removed), 3),
        **scores,
    }
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.float().contiguous() for k, v in vectors.items()}, str(cfg.result_dir / "vectors.safetensors"))
    (cfg.result_dir / "removal.json").write_text(json.dumps(summary, indent=1))
    return summary


def story_token_scores(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: ThemeRemovalRunConfig,
    vectors: dict[str, Tensor],
    texts: list[str],
) -> list[dict]:
    """Per story: its tokens and each token's cosine with d and d_minus (MLP output minus the train mean output)."""
    units = {k: (vectors[k] / vectors[k].norm()).float().to(model.device) for k in ("d", "d_minus")}
    mean_out = vectors["mean_out"].float().to(model.device)
    rows = []
    for start in range(0, len(texts), cfg.tokens.batch_size):
        batch = texts[start : start + cfg.tokens.batch_size]
        ids = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in batch]
        input_ids, mask = padded_batch(ids, tokenizer.pad_token_id)
        acts = read_layer(model, cfg.layer, input_ids.to(model.device), mask.to(model.device))
        out: Float[Tensor, "batch seq d_model"] = (acts["resid_post"] - acts["resid_mid"]).float() - mean_out
        out = out / out.norm(dim=-1, keepdim=True)
        for j, story_ids in enumerate(ids):
            n = len(story_ids)
            rows.append({
                "tokens": [tokenizer.decode([i]) for i in story_ids],
                **{k: (out[j, :n] @ u).tolist() for k, u in units.items()},
            })  # fmt: skip
    return rows


def heldout_texts(cfg: ThemeRemovalRunConfig) -> tuple[list[str], list[str], list[str]]:
    """Texts, labels and ids of the held-out stories, in run order."""
    stories = [s for s in load_stories(cfg.data) if s.heldout]
    return [s.text for s in stories], [s.emotion for s in stories], [s.id for s in stories]
