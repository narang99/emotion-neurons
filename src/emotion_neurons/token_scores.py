"""Per-token scores against saved vectors, for the token views of the reports. GPU.

- Stories: every held-out story (12 emotions and neutral), each token scored against all 12 emotion vectors.
  - Score: cosine of the token's activation, minus the grand mean, with neutral PCs removed, with each denoised vector.
  - This is the nearest-vector check of `vectors.checks`, done per token instead of per story mean.
- Direction: every val prompt, followed by its saved plain reply, projected onto the unit direction at its layer's input.
  - Plain replies only: the ablated and added runs change the stream, so their tokens would need those hooks too.
- Saved as results/<run_id>/token_scores.json (texts, tokens, offsets) and token_scores.safetensors (scores, fp16).
"""

import json
from pathlib import Path

import torch
from jaxtyping import Float
from pydantic import BaseModel
from safetensors.torch import load_file, save_file
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from .config import DirectionRunConfig, VectorsRunConfig
from .datasets import InstructionSet
from .emotions import EMOTIONS
from .models import adapter_for, load_model, read_layer, read_resid_pre
from .texts import load_stories
from .vectors import remove

Scores = Float[Tensor, "token series"]


class TokenText(BaseModel):
    """One scored text: its tokens and where its scores sit in the flat score array."""

    id: str
    label: str
    tokens: list[str]
    start: int
    marker: int | None = None
    """For a prompt with its reply: the first reply token."""


class TokenScores(BaseModel):
    series: list[str]
    texts: list[TokenText]


def token_strings(tokenizer: PreTrainedTokenizerBase, ids: list[int]) -> list[str]:
    return [tokenizer.decode([i]) for i in ids]


def padded(rows: list[list[int]], pad_id: int) -> tuple[Tensor, Tensor]:
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for j, r in enumerate(rows):
        ids[j, : len(r)] = torch.tensor(r)
        mask[j, : len(r)] = 1
    return ids, mask


def emotion_cosines(
    x: Float[Tensor, "seq dim"], grand: Float[Tensor, " dim"], pcs: Float[Tensor, "k dim"], vectors: Float[Tensor, "emotion dim"]
) -> Float[Tensor, "seq emotion"]:
    centred = remove(x - grand, pcs)
    return torch.nn.functional.normalize(centred, dim=-1) @ torch.nn.functional.normalize(vectors, dim=-1).T


def site_activations(model, layer: int, ids: Tensor, mask: Tensor, site: str) -> Tensor:
    acts = read_layer(model, layer, ids, mask)
    return acts["resid_post"] - acts["resid_mid"] if site == "mlp_out" else acts[site]


def story_token_scores(cfg: VectorsRunConfig) -> tuple[TokenScores, Scores]:
    site = cfg.tokens.site
    saved = load_file(str(cfg.result_dir / "vectors.safetensors"))
    vectors, grand, pcs = (saved[f"{site}/{k}"].double() for k in ("mean/denoised", "grand_mean", "neutral_pcs"))
    stories = [s for s in load_stories(cfg.data) if s.heldout]
    model, tokenizer = load_model(cfg.model, cfg.device)
    vectors, grand, pcs = vectors.to(model.device), grand.to(model.device), pcs.to(model.device)
    texts, rows, start = [], [], 0
    for b in range(0, len(stories), cfg.tokens.batch_size):
        batch = stories[b : b + cfg.tokens.batch_size]
        encoded = [tokenizer(s.text)["input_ids"] for s in batch]
        ids, mask = padded(encoded, tokenizer.pad_token_id)
        x = site_activations(model, cfg.layer, ids.to(model.device), mask.to(model.device), site).double()
        for j, (story, story_ids) in enumerate(zip(batch, encoded)):
            rows.append(emotion_cosines(x[j, : len(story_ids)], grand, pcs, vectors).float().cpu())
            texts.append(TokenText(id=story.id, label=story.emotion, tokens=token_strings(tokenizer, story_ids), start=start))
            start += len(story_ids)
        print(f"[tokens] {min(b + cfg.tokens.batch_size, len(stories))} of {len(stories)} stories", flush=True)
    return TokenScores(series=list(EMOTIONS), texts=texts), torch.cat(rows)


def direction_token_scores(cfg: DirectionRunConfig) -> tuple[TokenScores, Scores]:
    choice = json.loads((cfg.result_dir / "direction.json").read_text())["choice"]
    replies = json.loads((cfg.result_dir / "replies.json").read_text())
    unit = torch.nn.functional.normalize(load_file(str(cfg.result_dir / "direction.safetensors"))["direction"].double(), dim=0)
    adapter = adapter_for(cfg.model)
    instructions = InstructionSet(cfg.instructions)
    model, tokenizer = load_model(cfg.model, cfg.device)
    unit = unit.to(model.device)
    items = []
    for category in cfg.instructions.categories:
        val = [e for e in instructions.examples() if e.split == "val" and e.category == category]
        for example, reply in zip(val, replies[f"{category}/plain"], strict=True):
            prompt = instructions.render(example, tokenizer, adapter).input_ids
            items.append((example, prompt, prompt + tokenizer(reply, add_special_tokens=False)["input_ids"]))
    texts, rows, start = [], [], 0
    for b in range(0, len(items), cfg.tokens.batch_size):
        batch = items[b : b + cfg.tokens.batch_size]
        ids, mask = padded([full for _, _, full in batch], tokenizer.pad_token_id)
        x = read_resid_pre(model, choice["layer"], ids.to(model.device), mask.to(model.device)).double()
        for j, (example, prompt, full) in enumerate(batch):
            rows.append((x[j, : len(full)] @ unit)[:, None].float().cpu())
            texts.append(TokenText(id=example.id, label=example.category, tokens=token_strings(tokenizer, full), start=start, marker=len(prompt)))
            start += len(full)
        print(f"[tokens] {min(b + cfg.tokens.batch_size, len(items))} of {len(items)} prompts", flush=True)
    return TokenScores(series=["projection"], texts=texts), torch.cat(rows)


def save_token_scores(result_dir: Path, scores: TokenScores, values: Scores) -> None:
    assert values.shape == (sum(len(t.tokens) for t in scores.texts), len(scores.series)), "scores do not match the texts"
    assert torch.isfinite(values).all(), "non-finite token scores"
    (result_dir / "token_scores.json").write_text(scores.model_dump_json())
    save_file({"scores": values.half().contiguous()}, str(result_dir / "token_scores.safetensors"))


def load_token_scores(result_dir: Path) -> tuple[TokenScores, Scores] | None:
    """None if this run has no token scores yet."""
    if not (result_dir / "token_scores.json").exists():
        return None
    scores = TokenScores.model_validate_json((result_dir / "token_scores.json").read_text())
    return scores, load_file(str(result_dir / "token_scores.safetensors"))["scores"].float()
