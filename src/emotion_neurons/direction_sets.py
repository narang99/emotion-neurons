"""The sparsest set of earlier MLP neurons behind a direction at one site, and how well the direction classifies
prompts once their parts are removed from it.

- Site: resid_pre of `layer`, at one post-instruction position (layer 18, last token, for refusal in Qwen-1 7B).
- Direction r: the target category's mean minus the other category's at the site, over the train prompts.
- A neuron's part of r is `u_n = Δā_n w_n`.
  - Δā_n: its mean activation at the site's position, target minus other, over the train prompts.
  - w_n: its column of W_down.
  - Only the direct write at the site's position counts, not paths through attention or later MLPs.
  - The sum of all parts is the summed write of the MLPs of layers 0 to layer - 1 into r.
  - The embedding and the attention layers write the rest of r.
- A mean ablation sets a neuron to one value for both categories, whatever the mean (band or all tokens).
  - So it removes exactly `u_n` from r.
- Orders (`config.NeuronOrder`), a union on `kind`:
  - greedy (`sparse.greedy_set`): the neurons whose parts rebuild r, most useful first;
  - random: the whole pool shuffled, the control.
- Reduced direction: r minus the parts of the first N neurons of an order.
- Accuracy: a prompt is called target when its cosine with the reduced direction is above a threshold.
  - The threshold is fitted on the train prompts, then applied to the val prompts.
  - The prompts' own activations are the intact model's: only the direction changes.
"""

from collections.abc import Iterator
from typing import NamedTuple

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel
from torch import Tensor
from transformers import PreTrainedModel

from .capture import padded
from .config import GreedyOrder, NeuronOrder, RandomOrder
from .datasets import Rendered
from .models import ModelAdapter, read_resid_pre
from .models.qwen3 import hooks, store_input
from .sparse import GreedyRun, greedy_set

Vector = Float[Tensor, " d_model"]
Parts = Float[Tensor, "d_model neuron"]
Scores = Float[Tensor, " prompt"]
Labels = Bool[Tensor, " prompt"]


class SiteCapture(NamedTuple):
    resid: Float[Tensor, "prompt d_model"]
    acts: Float[Tensor, "prompt layer neuron"]


class Ordering(NamedTuple):
    name: str
    order: Int[Tensor, " neuron"]
    greedy: GreedyRun | None


class CurvePoint(BaseModel):
    order: str
    removed: int
    rebuilt: float
    """Share of r that the removed parts rebuild."""
    cosine_with_direction: float
    train_accuracy: float
    val_accuracy: float


@torch.no_grad()
def capture_site(
    model: PreTrainedModel,
    adapter: ModelAdapter,
    rendered: list[Rendered],
    layer: int,
    position: int,
    pad_id: int,
    batch_size: int,
) -> SiteCapture:
    """resid_pre of `layer` and the down_proj inputs of every earlier layer, at each prompt's `position`, on the CPU."""
    store: dict[int, Tensor] = {}
    handles = [adapter.mlp(model, i).down_proj.register_forward_pre_hook(store_input(store, i)) for i in range(layer)]
    resid, acts = [], []
    with hooks(handles):
        for start in range(0, len(rendered), batch_size):
            chunk = rendered[start : start + batch_size]
            ids, mask = padded([r.input_ids for r in chunk], pad_id)
            x = read_resid_pre(model, layer, ids.to(model.device), mask.to(model.device))
            for j, r in enumerate(chunk):
                p = r.positions[position]
                resid.append(x[j, p].double().cpu())
                acts.append(torch.stack([store[i][j, p] for i in range(layer)]).float().cpu())
    return SiteCapture(resid=torch.stack(resid), acts=torch.stack(acts))


def w_down(model: PreTrainedModel, adapter: ModelAdapter, layers: int) -> Float[Tensor, "layer d_model neuron"]:
    return torch.stack([adapter.mlp(model, i).down_proj.weight.float().cpu() for i in range(layers)])


def neuron_parts(
    target_acts: Float[Tensor, "prompt layer neuron"],
    other_acts: Float[Tensor, "prompt layer neuron"],
    W: Float[Tensor, "layer d_model neuron"],
) -> Parts:
    """`u_n = Δā_n w_n` for every neuron, flattened as layer * d_mlp + neuron."""
    shift = (target_acts.double().mean(0) - other_acts.double().mean(0)).float()
    return rearrange(W * shift[:, None, :], "layer d_model neuron -> d_model (layer neuron)")


def neuron_order(spec: NeuronOrder, parts: Parts, direction: Vector) -> Ordering:
    match spec:
        case GreedyOrder(max_steps=steps):
            run = greedy_set(parts, direction.float().to(parts.device), steps)
            return Ordering(spec.name, run.order, run)
        case RandomOrder(seed=seed):
            order = torch.randperm(parts.shape[1], generator=torch.Generator().manual_seed(seed))
            return Ordering(spec.name, order, None)


def curve_sizes(sizes: list[int], length: int) -> list[int]:
    return sorted({0, length, *(s for s in sizes if s <= length)})


def cosines(resid: Float[Tensor, "prompt d_model"], direction: Vector) -> Scores:
    return F.cosine_similarity(resid, direction[None], dim=-1)


def fitted_threshold(scores: Scores, is_target: Labels) -> float:
    """The cut between neighbouring train scores with the best accuracy; above it means target."""
    s = scores.sort().values
    cuts = torch.cat([s[:1] - 1, (s[1:] + s[:-1]) / 2, s[-1:] + 1])
    hits = ((scores[None, :] > cuts[:, None]) == is_target[None, :]).double().mean(1)
    return float(cuts[hits.argmax()])


def accuracy(scores: Scores, is_target: Labels, threshold: float) -> float:
    return float(((scores > threshold) == is_target).double().mean())


def rebuilt_share(direction: Vector, removed: Vector) -> float:
    missing = direction - removed
    return 1 - float(missing @ missing) / float(direction @ direction)


def removed_sums(
    parts: Parts, order: Int[Tensor, " neuron"], sizes: list[int], chunk: int = 16384
) -> Iterator[tuple[int, Vector]]:
    """The summed parts of the first N neurons of `order`, for each N in `sizes` (ascending), built up in chunks."""
    total = torch.zeros(parts.shape[0], dtype=torch.float64, device=parts.device)
    done = 0
    for size in sizes:
        for start in range(done, size, chunk):
            idx = order[start : min(start + chunk, size)].to(parts.device)
            total += parts[:, idx].double().sum(1)
        done = size
        yield size, total.clone()


def curve(
    name: str,
    parts: Parts,
    order: Int[Tensor, " neuron"],
    sizes: list[int],
    direction: Vector,
    train: tuple[Float[Tensor, "prompt d_model"], Labels],
    val: tuple[Float[Tensor, "prompt d_model"], Labels],
) -> list[CurvePoint]:
    points = []
    for size, removed in removed_sums(parts, order, sizes):
        removed = removed.cpu()
        reduced = direction - removed
        threshold = fitted_threshold(cosines(train[0], reduced), train[1])
        points.append(
            CurvePoint(
                order=name,
                removed=size,
                rebuilt=rebuilt_share(direction, removed),
                cosine_with_direction=float(F.cosine_similarity(reduced, direction, dim=0)),
                train_accuracy=accuracy(cosines(train[0], reduced), train[1], threshold),
                val_accuracy=accuracy(cosines(val[0], reduced), val[1], threshold),
            )
        )
    return points


def unit_name(index: int, d_mlp: int) -> dict[str, int]:
    return {"layer": index // d_mlp, "neuron": index % d_mlp}
