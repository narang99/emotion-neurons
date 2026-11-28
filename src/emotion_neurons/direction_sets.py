"""The sparsest set of MLP neurons behind a direction at one site, and how well the direction classifies prompts
once their parts are removed from it.

- Sites (`config.DirectionSite`), a union on `kind`, each at one post-instruction position:
  - resid_pre of a layer: the pool is every neuron of the earlier MLPs; attention and the embedding write the rest;
  - mlp_out of a layer: its down_proj output; the pool is that MLP's neurons, and their parts sum to all of r.
- Direction r: the target category's mean minus the other category's at the site, over the train prompts.
- A neuron's part of r is `u_n = Δā_n w_n`.
  - Δā_n: its mean activation at the site's position, target minus other, over the train prompts.
  - w_n: its column of W_down.
  - Only the direct write at the site's position counts, not paths through attention or later MLPs.
- A mean ablation sets a neuron to one value for both categories, whatever the mean (band or all tokens).
  - So it removes exactly `u_n` from r.
- Orders (`config.NeuronOrder`), a union on `kind`:
  - greedy (`sparse.greedy_set`): the neurons whose parts rebuild r, most useful first;
  - random: the whole pool shuffled, the control.
- For the first k neurons of an order, evaluations (`config.Evaluation`), a union on `kind`:
  - probe: the reduced direction r - sum u_n scores the intact prompts by cosine;
  - ablation: the k neurons are mean-ablated in every prompt too, then scored by the reduced direction;
  - only: each prompt keeps only the k neurons' write, scored by projection on their parts.
- Accuracy: a prompt is called target when its score is above a threshold.
  - The threshold is fitted on the train prompts, then applied to the val prompts.
- Random directions: random unit vectors score the intact prompts by cosine, the floor of the probe.
- Band subsets: random neuron sets split into the neurons that stay in their band and those that reach a tail,
  each part scored by `only`, against random drops of the same size.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel
from safetensors.torch import load_file
from torch import Tensor
from transformers import PreTrainedModel

from .capture import padded
from .config import (
    AblationEvaluation,
    BandSubsets,
    DirectionSite,
    Evaluation,
    GreedyOrder,
    MlpOutSite,
    NeuronOrder,
    OnlyEvaluation,
    ProbeEvaluation,
    RandomDirections,
    RandomOrder,
    ResidPreSite,
)
from .datasets import Rendered
from .models import ModelAdapter, read_resid_pre
from .models.qwen3 import hooks, store_input
from .sparse import GreedyRun, greedy_set

Vector = Float[Tensor, " d_model"]
Parts = Float[Tensor, "d_model neuron"]
Scores = Float[Tensor, " prompt"]
Labels = Bool[Tensor, " prompt"]
Prompts = Float[Tensor, "prompt d_model"]


class SiteCapture(NamedTuple):
    resid: Prompts
    acts: Float[Tensor, "prompt layer neuron"]


class Ordering(NamedTuple):
    name: str
    order: Int[Tensor, " neuron"]
    greedy: GreedyRun | None


class Pool(NamedTuple):
    """The pool's tensors on one device, neurons flattened as layer * d_mlp + neuron."""

    parts: Parts
    W: Float[Tensor, "d_model neuron"]
    train_acts: Float[Tensor, "prompt neuron"]
    val_acts: Float[Tensor, "prompt neuron"]
    means: Float[Tensor, " neuron"] | None


class Accumulated(NamedTuple):
    """Sums over the first `size` neurons of an order, fp64 on the CPU."""

    size: int
    parts: Vector
    train_writes: Prompts
    val_writes: Prompts
    mean_write: Vector | None


class CurvePoint(BaseModel):
    order: str
    evaluation: str
    neurons: int
    rebuilt: float
    """Share of r that the first `neurons` parts rebuild."""
    cosine_with_direction: float
    """Of the direction that scores the prompts."""
    train_accuracy: float
    val_accuracy: float


@torch.no_grad()
def capture_site(
    model: PreTrainedModel,
    adapter: ModelAdapter,
    rendered: list[Rendered],
    site: DirectionSite,
    position: int,
    pad_id: int,
    batch_size: int,
) -> SiteCapture:
    """resid_pre of the site's stop layer and the down_proj inputs of its pool layers, at each prompt's `position`."""
    store: dict[int, Tensor] = {}
    layers = site.pool_layers
    handles = [adapter.mlp(model, i).down_proj.register_forward_pre_hook(store_input(store, i)) for i in layers]
    resid, acts = [], []
    with hooks(handles):
        for start in range(0, len(rendered), batch_size):
            chunk = rendered[start : start + batch_size]
            ids, mask = padded([r.input_ids for r in chunk], pad_id)
            x = read_resid_pre(model, site.stop_layer, ids.to(model.device), mask.to(model.device))
            for j, r in enumerate(chunk):
                p = r.positions[position]
                resid.append(x[j, p].double().cpu())
                acts.append(torch.stack([store[i][j, p] for i in layers]).float().cpu())
    return SiteCapture(resid=torch.stack(resid), acts=torch.stack(acts))


def w_down(model: PreTrainedModel, adapter: ModelAdapter, layers: list[int]) -> Float[Tensor, "layer d_model neuron"]:
    return torch.stack([adapter.mlp(model, i).down_proj.weight.float().cpu() for i in layers])


def site_vectors(site: DirectionSite, capture: SiteCapture, W: Float[Tensor, "layer d_model neuron"]) -> Prompts:
    """Each prompt's vector at the site, in fp64."""
    match site:
        case ResidPreSite():
            return capture.resid
        case MlpOutSite():
            return einsum(W[0].double(), capture.acts[:, 0].double(), "d_model neuron, prompt neuron -> prompt d_model")


def neuron_parts(
    target_acts: Float[Tensor, "prompt layer neuron"],
    other_acts: Float[Tensor, "prompt layer neuron"],
    W: Float[Tensor, "layer d_model neuron"],
) -> Parts:
    """`u_n = Δā_n w_n` for every neuron in fp64, flattened as layer * d_mlp + neuron."""
    shift = target_acts.double().mean(0) - other_acts.double().mean(0)
    return rearrange(W.double() * shift[:, None, :], "layer d_model neuron -> d_model (layer neuron)")


def neuron_order(spec: NeuronOrder, parts: Parts, direction: Vector) -> Ordering:
    match spec:
        case GreedyOrder(max_steps=steps):
            run = greedy_set(parts.float(), direction.float().to(parts.device), steps)
            return Ordering(spec.name, run.order, run)
        case RandomOrder(seed=seed):
            order = torch.randperm(parts.shape[1], generator=torch.Generator().manual_seed(seed))
            return Ordering(spec.name, order, None)


def curve_sizes(sizes: list[int], length: int) -> list[int]:
    return sorted({0, length, *(s for s in sizes if s <= length)})


def cosines(vectors: Prompts, direction: Vector) -> Scores:
    return F.cosine_similarity(vectors, direction[None], dim=-1)


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


def flat_acts(acts: Float[Tensor, "prompt layer neuron"]) -> Float[Tensor, "prompt neuron"]:
    return rearrange(acts, "prompt layer neuron -> prompt (layer neuron)")


def flat_w(W: Float[Tensor, "layer d_model neuron"]) -> Float[Tensor, "d_model neuron"]:
    return rearrange(W, "layer d_model neuron -> d_model (layer neuron)")


def pool_means(path: Path, layers: list[int]) -> Float[Tensor, " neuron"]:
    """The `mean` of each pool layer from a poles.py thresholds file, flattened like the pool."""
    t = load_file(str(path))
    rows = [t["layers"].tolist().index(layer) for layer in layers]
    return t["mean"][rows].flatten()


def accumulated(pool: Pool, order: Int[Tensor, " neuron"], sizes: list[int], chunk: int = 4096) -> Iterator[Accumulated]:
    """For each size in `sizes` (ascending): the summed parts, each prompt's summed write `sum a_n w_n`, and the
    summed mean write `sum m_n w_n`, over the first `size` neurons of `order`."""
    device, d_model = pool.parts.device, pool.parts.shape[0]
    parts = torch.zeros(d_model, dtype=torch.float64, device=device)
    train = torch.zeros(pool.train_acts.shape[0], d_model, dtype=torch.float64, device=device)
    val = torch.zeros(pool.val_acts.shape[0], d_model, dtype=torch.float64, device=device)
    mean = torch.zeros(d_model, dtype=torch.float64, device=device)
    done = 0
    for size in sizes:
        for start in range(done, size, chunk):
            idx = order[start : min(start + chunk, size)].to(device)
            W = pool.W[:, idx].double()
            parts += pool.parts[:, idx].sum(1)
            train += pool.train_acts[:, idx].double() @ W.T
            val += pool.val_acts[:, idx].double() @ W.T
            if pool.means is not None:
                mean += W @ pool.means[idx].double()
        done = size
        yield Accumulated(
            size=size,
            parts=parts.cpu(),
            train_writes=train.cpu(),
            val_writes=val.cpu(),
            mean_write=mean.cpu() if pool.means is not None else None,
        )


def scored(
    evaluation: Evaluation, acc: Accumulated, direction: Vector, train: Prompts, val: Prompts
) -> tuple[Vector, Scores, Scores]:
    """(the scoring direction, train scores, val scores) of one evaluation."""
    match evaluation:
        case ProbeEvaluation():
            reduced = direction - acc.parts
            return reduced, cosines(train, reduced), cosines(val, reduced)
        case AblationEvaluation():
            assert acc.mean_write is not None, "ablation needs the pool's means"
            reduced = direction - acc.parts
            ablated_train = train - acc.train_writes + acc.mean_write
            ablated_val = val - acc.val_writes + acc.mean_write
            return reduced, cosines(ablated_train, reduced), cosines(ablated_val, reduced)
        case OnlyEvaluation():
            return acc.parts, acc.train_writes @ acc.parts, acc.val_writes @ acc.parts


def curve(
    name: str,
    pool: Pool,
    order: Int[Tensor, " neuron"],
    sizes: list[int],
    evaluations: list[Evaluation],
    direction: Vector,
    train: tuple[Prompts, Labels],
    val: tuple[Prompts, Labels],
) -> list[CurvePoint]:
    points = []
    for acc in accumulated(pool, order, sizes):
        for evaluation in evaluations:
            vector, train_scores, val_scores = scored(evaluation, acc, direction, train[0], val[0])
            threshold = fitted_threshold(train_scores, train[1])
            points.append(
                CurvePoint(
                    order=name,
                    evaluation=evaluation.kind,
                    neurons=acc.size,
                    rebuilt=rebuilt_share(direction, acc.parts),
                    cosine_with_direction=float(F.cosine_similarity(vector, direction, dim=0)),
                    train_accuracy=accuracy(train_scores, train[1], threshold),
                    val_accuracy=accuracy(val_scores, val[1], threshold),
                )
            )
    return points


class RandomBaseline(BaseModel):
    count: int
    val_accuracies: list[float]
    mean: float
    quantiles: dict[str, float]


def random_baseline(spec: RandomDirections, train: tuple[Prompts, Labels], val: tuple[Prompts, Labels]) -> RandomBaseline:
    g = torch.Generator().manual_seed(spec.seed)
    accuracies = []
    for _ in range(spec.count):
        v = torch.randn(train[0].shape[1], generator=g, dtype=torch.float64)
        threshold = fitted_threshold(cosines(train[0], v), train[1])
        accuracies.append(accuracy(cosines(val[0], v), val[1], threshold))
    a = torch.tensor(accuracies)
    return RandomBaseline(
        count=spec.count,
        val_accuracies=accuracies,
        mean=float(a.mean()),
        quantiles={str(q): float(a.quantile(q)) for q in (0.0, 0.1, 0.5, 0.9, 1.0)},
    )


def pool_poles(path: Path, low: str, high: str, layers: list[int]) -> tuple[Float[Tensor, " neuron"], Float[Tensor, " neuron"]]:
    """The low and high pole of each pool layer from a poles.py thresholds file, flattened like the pool."""
    t = load_file(str(path))
    rows = [t["layers"].tolist().index(layer) for layer in layers]
    return t[low][rows].flatten(), t[high][rows].flatten()


def tail_rates(
    acts: Float[Tensor, "prompt neuron"], low: Float[Tensor, " neuron"], high: Float[Tensor, " neuron"]
) -> Float[Tensor, " neuron"]:
    """Share of prompts where each neuron is at or beyond a pole, the complement of `ablation.band_mask`."""
    return ((acts <= low) | (acts >= high)).double().mean(0)


def only_accuracy(
    pool: Pool, neurons: Int[Tensor, " k"], direction: Vector, train: tuple[Prompts, Labels], val: tuple[Prompts, Labels]
) -> float | None:
    """Val accuracy of the `only` evaluation with exactly these neurons; None for an empty set."""
    if len(neurons) == 0:
        return None
    acc = next(accumulated(pool, neurons, [len(neurons)]))
    _, train_scores, val_scores = scored(OnlyEvaluation(), acc, direction, train[0], val[0])
    return accuracy(val_scores, val[1], fitted_threshold(train_scores, train[1]))


class BandSubset(BaseModel):
    neurons: list[int]
    tail_rates: list[float]
    projections: list[float]
    """Each neuron's part projected on the unit direction, `u_n . r / |r|`."""
    part_norms: list[float]
    projection_ranks: list[int]
    """Rank by projection among the whole pool, 1 for the largest."""
    greedy_ranks: list[int | None]
    """Position in the greedy order, 1 for the first; None when greedy never added the neuron."""
    in_band: list[int]
    all_accuracy: float
    in_band_accuracy: float | None
    tail_accuracy: float | None
    drop_accuracies: list[float]
    """Val accuracy after dropping as many random neurons as the tail part, one entry per draw."""


class NeuronRanks(NamedTuple):
    """Per-neuron numbers of the whole pool, used to describe any subset."""

    projections: Float[Tensor, " neuron"]
    part_norms: Float[Tensor, " neuron"]
    projection_ranks: Int[Tensor, " neuron"]
    greedy_ranks: Int[Tensor, " neuron"]
    """0 when greedy never added the neuron."""


def neuron_ranks(parts: Parts, direction: Vector, greedy_order: Int[Tensor, " step"]) -> NeuronRanks:
    projections = (parts.T @ (direction / direction.norm()).to(parts)).cpu()
    projection_ranks = torch.empty_like(projections, dtype=torch.long)
    projection_ranks[projections.argsort(descending=True)] = torch.arange(1, len(projections) + 1)
    greedy_ranks = torch.zeros(len(projections), dtype=torch.long)
    greedy_ranks[greedy_order] = torch.arange(1, len(greedy_order) + 1)
    return NeuronRanks(projections, parts.norm(dim=0).cpu(), projection_ranks, greedy_ranks)


def band_subset(
    spec: BandSubsets,
    neurons: Int[Tensor, " k"],
    rates: Float[Tensor, " neuron"],
    ranks: NeuronRanks,
    pool: Pool,
    direction: Vector,
    train: tuple[Prompts, Labels],
    val: tuple[Prompts, Labels],
    g: torch.Generator,
) -> BandSubset:
    in_band = rates[neurons] <= spec.max_tail_rate
    kept = int(in_band.sum())
    drops = [neurons[torch.randperm(len(neurons), generator=g)[:kept]] for _ in range(spec.drops)] if kept else []
    return BandSubset(
        neurons=neurons.tolist(),
        tail_rates=rates[neurons].tolist(),
        projections=ranks.projections[neurons].tolist(),
        part_norms=ranks.part_norms[neurons].tolist(),
        projection_ranks=ranks.projection_ranks[neurons].tolist(),
        greedy_ranks=[int(r) or None for r in ranks.greedy_ranks[neurons]],
        in_band=neurons[in_band].tolist(),
        all_accuracy=only_accuracy(pool, neurons, direction, train, val),
        in_band_accuracy=only_accuracy(pool, neurons[in_band], direction, train, val),
        tail_accuracy=only_accuracy(pool, neurons[~in_band], direction, train, val),
        drop_accuracies=[only_accuracy(pool, d, direction, train, val) for d in drops],
    )


def band_subsets(
    spec: BandSubsets,
    first_set: Int[Tensor, " k"],
    rates: Float[Tensor, " neuron"],
    ranks: NeuronRanks,
    pool: Pool,
    direction: Vector,
    train: tuple[Prompts, Labels],
    val: tuple[Prompts, Labels],
) -> list[BandSubset]:
    """`first_set`, then `spec.sets - 1` more random sets of the same size, drawn with seed + 1 so they differ from it."""
    g = torch.Generator().manual_seed(spec.seed + 1)
    n = pool.parts.shape[1]
    sets = [first_set] + [torch.randperm(n, generator=g)[: spec.set_size] for _ in range(spec.sets - 1)]
    return [band_subset(spec, s, rates, ranks, pool, direction, train, val, g) for s in sets]


def unit_name(index: int, layers: list[int], d_mlp: int) -> dict[str, int]:
    return {"layer": layers[index // d_mlp], "neuron": index % d_mlp}
