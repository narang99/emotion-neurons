"""Per-token scores against saved vectors, for the token views of the reports. GPU.

- Stories: every held-out story (12 emotions and neutral), each token scored against all 12 emotion vectors.
  - Score: cosine of the token's raw activation with each denoised vector, as in the emotions paper.
  - The paper removes the neutral PCs from the vectors only. Removing them from the activation too would change only
    the cosine's norm, since the denoised vectors have no component along them.
  - Unlike the held-out accuracy in `vectors.checks`, nothing is centred.
- Direction: every val prompt, followed by its saved plain reply, projected onto the unit direction at its layer's input.
  - Plain replies only: the ablated and added runs change the stream, so their tokens would need those hooks too.
- Layout: one flat `Scores` tensor for all texts, and a `ScoreIndex` that says which rows belong to which text.
  - `scores` [token, probe]: the texts' tokens one after another, a row per token, a column per probe.
  - A probe is one vector the tokens are scored against: each of the 12 emotion vectors, or the refusal direction.
  - A text's rows are `scores[start : start + len(tokens)]`, in token order (`reports.token_views.scores_of`).
  - `start` is the running token count of the texts before it, so the entries tile the tensor with no gaps.
- Saved as results/<run_id>/token_scores.json (the index) and token_scores.safetensors (`scores`, fp16).
"""

import json
from pathlib import Path

import torch
from jaxtyping import Float
from pydantic import BaseModel
from safetensors.torch import load_file, save_file
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .capture import padded
from .config import ExtractionRunConfig
from .datasets import InstructionSet
from .emotions import EMOTIONS
from .models import ModelAdapter, read_layer, read_resid_pre
from .texts import load_stories

Scores = Float[Tensor, "token probe"]
StoryScores = Float[Tensor, "story probe"]


class TokenText(BaseModel):
    """One scored text: its tokens and where its rows sit in the flat `Scores` tensor."""

    id: str
    label: str
    """The story's emotion (or neutral), or the prompt's category."""
    tokens: list[str]
    """Each token decoded on its own, for display."""
    start: int
    """Row of this text's first token in the flat `Scores` tensor."""
    marker: int | None = None
    """Prompts with their replies only: the index of the first reply token within `tokens`, where the prompt ends."""


class ScoreIndex(BaseModel):
    """Which rows of the flat `Scores` tensor belong to which text. It holds no scores itself."""

    probes: list[str]
    """The columns of the `Scores` tensor: the 12 emotions, or a single `projection` for a direction."""
    texts: list[TokenText]


def token_strings(tokenizer: PreTrainedTokenizerBase, ids: list[int]) -> list[str]:
    return [tokenizer.decode([i]) for i in ids]


def emotion_cosines(x: Float[Tensor, "seq dim"], vectors: Float[Tensor, "emotion dim"]) -> Float[Tensor, "seq emotion"]:
    return torch.nn.functional.normalize(x, dim=-1) @ torch.nn.functional.normalize(vectors, dim=-1).T


def site_activations(model, layer: int, ids: Tensor, mask: Tensor, site: str) -> Tensor:
    acts = read_layer(model, layer, ids, mask)
    return acts["resid_post"] - acts["resid_mid"] if site == "mlp_out" else acts[site]


def story_token_scores(
    cfg: ExtractionRunConfig, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
) -> tuple[ScoreIndex, Scores]:
    """Every held-out story, each token scored against the 12 denoised emotion vectors.

    - `Scores` is one flat [total_tokens, probe] tensor, with a column per emotion.
      - Each story's rows are `scores[start : start + len(tokens)]`.
    - `ScoreIndex.texts` has one `TokenText` per story, with its own tokens and `start`. `marker` stays None.
    """
    tokens = cfg.emotion_vectors.tokens
    site = tokens.site
    saved = load_file(str(cfg.result_dir / "vectors.safetensors"))
    vectors = saved[f"{site}/mean/denoised"].double().to(model.device)
    stories = [s for s in load_stories(cfg.stories) if s.heldout]
    texts, rows, start = [], [], 0
    for b in range(0, len(stories), tokens.batch_size):
        batch = stories[b : b + tokens.batch_size]
        encoded = [tokenizer(s.text)["input_ids"] for s in batch]
        ids, mask = padded(encoded, tokenizer.pad_token_id)
        x = site_activations(model, cfg.emotion_vectors.extract.layer, ids.to(model.device), mask.to(model.device), site).double()
        for j, (story, story_ids) in enumerate(zip(batch, encoded)):
            rows.append(emotion_cosines(x[j, : len(story_ids)], vectors).float().cpu())
            texts.append(TokenText(id=story.id, label=story.emotion, tokens=token_strings(tokenizer, story_ids), start=start))
            start += len(story_ids)
        print(f"[tokens] {min(b + tokens.batch_size, len(stories))} of {len(stories)} stories", flush=True)
    return ScoreIndex(probes=list(EMOTIONS), texts=texts), torch.cat(rows)


def direction_token_scores(
    cfg: ExtractionRunConfig, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, adapter: ModelAdapter
) -> tuple[ScoreIndex, Scores]:
    """Every val prompt with its plain reply, each token projected onto the unit direction.

    - `Scores` is one flat [total_tokens, probe] tensor, with the single column `projection`.
      - Each prompt's rows are `scores[start : start + len(tokens)]`, prompt tokens first, then reply tokens.
    - `ScoreIndex.texts` has one `TokenText` per prompt, with its own tokens, `start` and `marker`.
      - `marker` is the index of the first reply token within `tokens`.
    """
    tokens = cfg.direction.tokens
    choice = json.loads((cfg.result_dir / "direction.json").read_text())["choice"]
    replies = json.loads((cfg.result_dir / "replies.json").read_text())
    unit = torch.nn.functional.normalize(load_file(str(cfg.result_dir / "direction.safetensors"))["direction"].double(), dim=0)
    instructions = InstructionSet(cfg.instructions)
    unit = unit.to(model.device)
    items = []
    for category in cfg.instructions.categories:
        val = [e for e in instructions.examples() if e.split == "val" and e.category == category]
        for example, reply in zip(val, replies[f"{category}/plain"], strict=True):
            prompt = instructions.render(example, tokenizer, adapter).input_ids
            items.append((example, prompt, prompt + tokenizer(reply, add_special_tokens=False)["input_ids"]))
    texts, rows, start = [], [], 0
    for b in range(0, len(items), tokens.batch_size):
        batch = items[b : b + tokens.batch_size]
        ids, mask = padded([full for _, _, full in batch], tokenizer.pad_token_id)
        x = read_resid_pre(model, choice["layer"], ids.to(model.device), mask.to(model.device)).double()
        for j, (example, prompt, full) in enumerate(batch):
            rows.append((x[j, : len(full)] @ unit)[:, None].float().cpu())
            texts.append(
                TokenText(
                    id=example.id, label=example.category, tokens=token_strings(tokenizer, full), start=start, marker=len(prompt)
                )
            )
            start += len(full)
        print(f"[tokens] {min(b + tokens.batch_size, len(items))} of {len(items)} prompts", flush=True)
    return ScoreIndex(probes=["projection"], texts=texts), torch.cat(rows)


def save_token_scores(result_dir: Path, index: ScoreIndex, scores: Scores) -> None:
    assert scores.shape == (sum(len(t.tokens) for t in index.texts), len(index.probes)), "scores do not match the texts"
    assert torch.isfinite(scores).all(), "non-finite token scores"
    (result_dir / "token_scores.json").write_text(index.model_dump_json())
    save_file({"scores": scores.half().contiguous()}, str(result_dir / "token_scores.safetensors"))


def load_token_scores(result_dir: Path) -> tuple[ScoreIndex, Scores] | None:
    """None if this run has no token scores yet."""
    if not (result_dir / "token_scores.json").exists():
        return None
    index = ScoreIndex.model_validate_json((result_dir / "token_scores.json").read_text())
    return index, load_file(str(result_dir / "token_scores.safetensors"))["scores"].float()
