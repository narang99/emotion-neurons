"""Report data for one difference-of-means direction (scripts/direction.py): read only from its saved results.

- Reads results/<run_id>/direction.json, direction.safetensors and replies.json. No model is loaded.
- Token views: from results/<run_id>/token_scores.*, if scripts/token_scores.py has run.
- The val instructions come from the config's sources (`datasets.InstructionSet`), in the order the replies were made.
- Selection grid: one candidate per post-instruction position and layer.
  - bypass: refusal score of target prompts with the candidate ablated (lower is better).
  - induce: refusal score of the other prompts with the candidate added (must be > 0).
  - kl: next-token KL of the other prompts when the candidate is ablated (must be < `selection.max_kl`).
"""

import json

import torch
from pydantic import BaseModel
from safetensors.torch import load_file

from ..config import DirectionRunConfig
from ..datasets import InstructionSet
from ..token_scores import load_token_scores
from .heatmap import Heatmap, heatmap
from .token_views import TokenView, prompt_views


class Choice(BaseModel):
    position: int
    layer: int
    bypass: float
    induce: float
    kl: float


class Condition(BaseModel):
    name: str
    description: str
    refusal_rate: float


class Reply(BaseModel):
    prompt: str
    reply: str
    refused: bool


class ReplyGroup(BaseModel):
    condition: str
    replies: list[Reply]


class DirectionReport(BaseModel):
    title: str
    model: str
    run_id: str
    target: str
    other: str
    baseline: str
    n_train: int
    n_val: int
    max_kl: float
    choice: Choice
    filtered_out_all: bool
    conditions: list[Condition]
    grid_maps: list[Heatmap]
    vector_maps: list[Heatmap]
    replies: list[ReplyGroup]
    token_views: list[TokenView]


def selection_maps(summary: dict, max_kl: float) -> list[Heatmap]:
    grid = summary["grid"]
    positions = sorted({g["position"] for g in grid})
    layers = list(range(min(g["layer"] for g in grid), max(g["layer"] for g in grid) + 1))
    at = {(g["position"], g["layer"]): g for g in grid}
    cell = {(positions.index(g["position"]), layers.index(g["layer"])) for g in grid}
    chosen = {(positions.index(summary["choice"]["position"]), layers.index(summary["choice"]["layer"]))}
    excluded = {
        (positions.index(p), layers.index(lay)) for (p, lay), g in at.items() if not (g["induce"] > 0 and g["kl"] < max_kl)
    }
    assert chosen <= cell, "the chosen candidate is not in the grid"

    def values(key: str) -> list[list[float | None]]:
        return [[at[(p, lay)][key] if (p, lay) in at else None for lay in layers] for p in positions]

    common = dict(row_labels=[str(p) for p in positions], col_labels=[str(lay) for lay in layers],
                  row_title="position", col_title="layer", marked=chosen, excluded=excluded, show_values=False)  # fmt: skip
    return [
        heatmap("bypass", values("bypass"), scale="diverging", caption="Refusal score of target prompts, candidate ablated. Lower is better.", **common),
        heatmap("induce", values("induce"), scale="diverging", caption="Refusal score of other prompts, candidate added. Must be above 0.", **common),
        heatmap("kl", values("kl"), scale="sequential", fmt=".3f", log=True, caption=f"Next-token KL of other prompts, candidate ablated. Must be below {max_kl}. Log colour scale.", **common),
    ]  # fmt: skip


def vector_maps(tensors: dict[str, torch.Tensor], choice: dict) -> list[Heatmap]:
    """Every candidate's norm and its cosine with the chosen direction, over all layers."""
    diffs = tensors["diffs"].double()
    n_pos, n_layers = diffs.shape[:2]
    positions = [p - n_pos for p in range(n_pos)]
    chosen = tensors["direction"].double()
    norms = diffs.norm(dim=-1)
    cosines = torch.nn.functional.cosine_similarity(diffs, chosen.expand_as(diffs), dim=-1)
    mark = {(positions.index(choice["position"]), choice["layer"])}
    common = dict(row_labels=[str(p) for p in positions], col_labels=[str(lay) for lay in range(n_layers)],
                  row_title="position", col_title="layer", marked=mark, show_values=False)  # fmt: skip
    return [
        heatmap("norm", norms.tolist(), scale="sequential", fmt=".1f", caption="Norm of the difference of means at the layer's input.", **common),
        heatmap("cosine with the chosen direction", cosines.tolist(), scale="diverging", caption="How far the chosen direction holds across layers and positions.", **common),
    ]  # fmt: skip


def is_refusal(reply: str, phrases: list[str]) -> bool:
    return any(p.lower() in reply.lower() for p in phrases)


def reply_groups(cfg: DirectionRunConfig, replies: dict[str, list[str]]) -> list[ReplyGroup]:
    examples = InstructionSet(cfg.instructions).examples()
    prompts = {c: [e.text for e in examples if e.split == "val" and e.category == c] for c in cfg.instructions.categories}
    groups = []
    for condition, texts in replies.items():
        category = condition.split("/")[0]
        assert len(prompts[category]) == len(texts), f"{condition}: {len(texts)} replies for {len(prompts[category])} prompts"
        rows = [
            Reply(prompt=p, reply=r, refused=is_refusal(r, cfg.generation.refusal_phrases))
            for p, r in zip(prompts[category], texts)
        ]
        groups.append(ReplyGroup(condition=condition, replies=rows))
    return groups


def conditions(summary: dict, target: str, other: str) -> list[Condition]:
    described = {
        f"{target}/plain": f"{target} prompts, no change",
        f"{target}/ablated": f"{target} prompts, direction ablated at every layer",
        f"{other}/plain": f"{other} prompts, no change",
        f"{other}/added": f"{other} prompts, direction added at its layer",
    }
    return [Condition(name=k, description=described[k], refusal_rate=v) for k, v in summary["refusal_rates"].items()]


def token_views(cfg: DirectionRunConfig) -> list[TokenView]:
    saved = load_token_scores(cfg.result_dir)
    if saved is None:
        return []
    return prompt_views(*saved, labels=list(cfg.instructions.categories), per_label=cfg.tokens.prompts_per_category)


def build_direction_report(cfg: DirectionRunConfig) -> DirectionReport:
    summary = json.loads((cfg.result_dir / "direction.json").read_text())
    replies = json.loads((cfg.result_dir / "replies.json").read_text())
    tensors = load_file(str(cfg.result_dir / "direction.safetensors"))
    target = cfg.direction.target
    other = next(c for c in cfg.instructions.categories if c != target)
    return DirectionReport(
        title=f"Direction: {target} minus {other}, {cfg.model}",
        model=cfg.model, run_id=cfg.run_id, target=target, other=other, baseline=cfg.direction.baseline,
        n_train=summary["n_train"], n_val=summary["n_val"], max_kl=cfg.selection.max_kl,
        choice=Choice(**summary["choice"]), filtered_out_all=summary["filtered_out_all"],
        conditions=conditions(summary, target, other),
        grid_maps=selection_maps(summary, cfg.selection.max_kl),
        vector_maps=vector_maps(tensors, summary["choice"]),
        replies=reply_groups(cfg, replies),
        token_views=token_views(cfg),
    )  # fmt: skip
