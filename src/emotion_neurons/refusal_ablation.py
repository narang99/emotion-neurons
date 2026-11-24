"""The refusal direction at one site, recomputed while the MLPs of the layers before it are mean-ablated.

- Site: the residual stream at one layer's input (resid_pre), at one post-instruction position (e.g. layer 18, -1).
- Direction: the target category's mean minus the other category's, over the train prompts (as in `directions`).
- A condition mean-ablates the MLP neurons of layers `start` to `layer - 1` (`ablation.MeanAblation`): a band of the
  poles, or the random control with as many neurons per token.
- Per condition:
  - the direction's cosine with the intact model's and its norm;
  - the ablated model's own refusal rate on the val prompts, by greedy generation and phrase matching;
  - the share of cells ablated, counted in the capture pass (generation passes are not counted).
- Cosine floor: directions from two random halves of the intact model's train prompts, over several splits.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import BaseModel
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .ablation import MeanAblation
from .capture import padded
from .config import RefusalCheckConfig
from .datasets import Rendered
from .directions import generate, refusal_rate
from .models import ModelAdapter, read_resid_pre

Activations = Float[Tensor, "prompt d_model"]
Vector = Float[Tensor, " d_model"]


class Condition(BaseModel):
    name: str
    band: str | None = None
    kind: str | None = None
    start: int | None = None
    layers: list[int] = []


class ConditionResult(BaseModel):
    condition: Condition
    ablated_fraction: float | None
    cosine: float
    norm: float
    refusal_rates: dict[str, float]


class SplitHalfFloor(BaseModel):
    cosines: list[float]
    mean: float
    min: float


@torch.no_grad()
def site_activations(
    model: PreTrainedModel,
    rendered: list[Rendered],
    layer: int,
    position: int,
    pad_id: int,
    batch_size: int,
    ablation: MeanAblation | None = None,
) -> Activations:
    """resid_pre of `layer` at each prompt's post-instruction `position`, fp64 on the CPU. The pass stops there."""
    rows = []
    for start in range(0, len(rendered), batch_size):
        chunk = rendered[start : start + batch_size]
        ids, mask = padded([r.input_ids for r in chunk], pad_id)
        ids, mask = ids.to(model.device), mask.to(model.device)
        if ablation is not None:
            ablation.tokens = mask.bool()
        x = read_resid_pre(model, layer, ids, mask)
        rows += [x[j, r.positions[position]].double().cpu() for j, r in enumerate(chunk)]
    if ablation is not None:
        ablation.tokens = None
    return torch.stack(rows)


def direction(target: Activations, other: Activations) -> Vector:
    return target.mean(0) - other.mean(0)


def cosine(a: Vector, b: Vector) -> float:
    return float(F.cosine_similarity(a, b, dim=0))


def split_half_floor(target: Activations, other: Activations, splits: int) -> SplitHalfFloor:
    """Cosine between the directions of two disjoint random halves of the prompts, per split."""
    cosines = []
    for seed in range(splits):
        g = torch.Generator().manual_seed(seed)
        t, o = torch.randperm(len(target), generator=g), torch.randperm(len(other), generator=g)
        a = direction(target[t[: len(t) // 2]], other[o[: len(o) // 2]])
        b = direction(target[t[len(t) // 2 :]], other[o[len(o) // 2 :]])
        cosines.append(cosine(a, b))
    return SplitHalfFloor(cosines=cosines, mean=sum(cosines) / len(cosines), min=min(cosines))


def refusal_replies(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, val: dict[str, list[list[int]]], g: RefusalCheckConfig
) -> dict[str, list[str]]:
    return {category: generate(model, tokenizer, prompts, g) for category, prompts in val.items()}


def run_condition(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    adapter: ModelAdapter,
    train: dict[str, list[Rendered]],
    val: dict[str, list[list[int]]],
    layer: int,
    position: int,
    batch_size: int,
    g: RefusalCheckConfig,
    ablation: MeanAblation | None,
) -> tuple[dict[str, Activations], dict[str, list[str]], float | None]:
    """(train activations at the site per category, val replies per category, ablated share), with `ablation` applied."""

    def capture() -> dict[str, Activations]:
        return {
            c: site_activations(model, r, layer, position, tokenizer.pad_token_id, batch_size, ablation) for c, r in train.items()
        }

    if ablation is None:
        acts, replies, fraction = capture(), refusal_replies(model, tokenizer, val, g), None
    else:
        with ablation.applied(model, adapter):
            acts = capture()
            fraction = sum(ablation.ablated_fraction().values()) / len(ablation.layers)
            replies = refusal_replies(model, tokenizer, val, g)
    return acts, replies, fraction


def site_vectors(acts: dict[str, Activations], target: str) -> dict[str, Vector]:
    """`<category>_mean` per category, and `direction`: the target's mean minus the other category's."""
    other = next(c for c in acts if c != target)
    return {f"{c}_mean": a.mean(0) for c, a in acts.items()} | {"direction": direction(acts[target], acts[other])}


def condition_result(
    condition: Condition,
    vectors: dict[str, Vector],
    intact: Vector,
    replies: dict[str, list[str]],
    fraction: float | None,
    phrases: list[str],
) -> ConditionResult:
    return ConditionResult(
        condition=condition,
        ablated_fraction=fraction,
        cosine=cosine(vectors["direction"], intact),
        norm=float(vectors["direction"].norm()),
        refusal_rates={c: refusal_rate(r, phrases) for c, r in replies.items()},
    )
