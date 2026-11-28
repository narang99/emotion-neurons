"""Mean ablation of MLP neurons, and what it does to the model's next-token predictions.

- Band: at every token, a neuron of an ablated layer whose value lies strictly between its low and high pole
  (e.g. p3 and p97 over general text, `poles`) gets a fixed replacement value instead. Only the tails stay.
  - The replacement is a stored tensor of the poles run: its mean over that text by default, or a quantile such as `p50`.
- Random control: at every token, as many neurons as the band ablates there, chosen uniformly at random, get their
  replacement values. The count comes from the band test on this run's own activations.
- Positions (`Ablation.positions`): every position may be ablated, or the first ones of each row stay intact.
  - The rule is applied before the random draw, so the control spares the same positions.
- The hook replaces the input of each ablated layer's `down_proj` (the adapter's `mlp`), so the change reaches the
  residual stream through the down projection, as a real activation would.
- Every layer's hook sees the activations that the ablations before it produce, as in the Goodfire paper.
- Against the intact model, on the same tokens:
  - loss: next-token cross-entropy, summed over the read positions;
  - KL(intact || ablated) of the next-token distributions, summed over the read positions.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel
from safetensors.torch import load_file
from torch import Tensor
from transformers import PreTrainedModel

from .capture.batch import padded, position_mask
from .config import AllPositions, PositionsConfig, SkipFirst
from .models import ModelAdapter
from .models.qwen3 import hooks

Neurons = Float[Tensor, "batch seq neuron"]
NeuronMask = Bool[Tensor, "batch seq neuron"]


class Poles(NamedTuple):
    layers: list[int]
    low: Float[Tensor, "layer neuron"]
    high: Float[Tensor, "layer neuron"]
    replacement: Float[Tensor, "layer neuron"]

    def row(self, layer: int) -> int:
        assert layer in self.layers, f"no poles for layer {layer}; the thresholds file has {self.layers}"
        return self.layers.index(layer)


def load_poles(path: Path, low: str, high: str, device: str | torch.device, replacement: str = "mean") -> Poles:
    t = load_file(str(path), device=str(device))
    return Poles(layers=t["layers"].tolist(), low=t[low], high=t[high], replacement=t[replacement])


def band_mask(a: Neurons, low: Float[Tensor, " neuron"], high: Float[Tensor, " neuron"]) -> NeuronMask:
    return (a > low) & (a < high)


def random_mask_like(mask: NeuronMask) -> NeuronMask:
    """As many True per token as `mask`, at uniformly random neurons: the k smallest of k random draws."""
    count = mask.sum(-1, keepdim=True)
    draws = torch.rand(mask.shape, device=mask.device)
    kth = draws.sort(-1).values.gather(-1, (count - 1).clamp(min=0))
    return (draws <= kth) & (count > 0)


def ablated_positions(positions: PositionsConfig, seq: int, device: torch.device) -> Bool[Tensor, " seq"]:
    """Rows are right padded (`capture.batch.padded`), so index i is the i-th token of every row."""
    match positions:
        case AllPositions():
            return torch.ones(seq, dtype=torch.bool, device=device)
        case SkipFirst(count=count):
            return torch.arange(seq, device=device) >= count


class MeanAblation:
    """The hooks of one ablation, and the share of cells they ablated at real tokens."""

    def __init__(
        self,
        poles: Poles,
        kind: Literal["band", "random"],
        layers: list[int],
        positions: PositionsConfig = AllPositions(),
    ):
        self.poles = poles
        self.kind = kind
        self.layers = layers
        self.positions = positions
        self.tokens: Bool[Tensor, "batch seq"] | None = None
        self.ablated = {layer: 0 for layer in layers}
        self.cells = {layer: 0 for layer in layers}

    def mask(self, a: Neurons, layer: int) -> NeuronMask:
        row = self.poles.row(layer)
        allowed = ablated_positions(self.positions, a.shape[1], a.device)
        mask = band_mask(a, self.poles.low[row], self.poles.high[row]) & allowed[:, None]
        return random_mask_like(mask) if self.kind == "random" else mask

    def replace(self, layer: int):
        replacement = self.poles.replacement[self.poles.row(layer)]

        def hook(_module, args):
            a = args[0]
            assert isinstance(self.positions, AllPositions) or self.counting(a), (
                "positions are counted from the start of a full pass; a cached generation step would skip the wrong ones"
            )
            mask = self.mask(a, layer)
            self.count(layer, mask)
            return (torch.where(mask, replacement.to(a.dtype), a),)

        return hook

    def counting(self, a: Neurons) -> bool:
        """True while `tokens` holds the attention mask of this full pass; e.g. `model.generate` steps are not counted."""
        return self.tokens is not None and self.tokens.shape == a.shape[:-1]

    def count(self, layer: int, mask: NeuronMask) -> None:
        if not self.counting(mask):
            return
        self.ablated[layer] += (mask & self.tokens[..., None]).sum()
        self.cells[layer] += self.tokens.sum() * mask.shape[-1]

    def reset_counts(self) -> None:
        self.ablated = {layer: 0 for layer in self.layers}
        self.cells = {layer: 0 for layer in self.layers}

    def ablated_fraction(self) -> dict[int, float]:
        return {layer: float(self.ablated[layer] / self.cells[layer]) for layer in self.layers}

    @contextmanager
    def applied(self, model: PreTrainedModel, adapter: ModelAdapter) -> Iterator[None]:
        handles = [adapter.mlp(model, layer).down_proj.register_forward_pre_hook(self.replace(layer)) for layer in self.layers]
        with hooks(handles):
            yield


class ReadBatch(NamedTuple):
    input_ids: Int[Tensor, "batch seq"]
    attention_mask: Int[Tensor, "batch seq"]
    read: Bool[Tensor, "batch seq"]
    """Positions whose next-token distribution counts."""


class Totals(BaseModel):
    positions: int = 0
    loss_intact: float = 0.0
    loss_ablated: float = 0.0
    kl: float = 0.0


def text_batches(
    windows: list[list[int]], pad_id: int, batch_size: int, skip_first: int, device: str | torch.device
) -> Iterator[ReadBatch]:
    """Every position from `skip_first` on that has a real next token."""
    for start in range(0, len(windows), batch_size):
        ids, mask = padded(windows[start : start + batch_size], pad_id)
        read = torch.zeros_like(mask, dtype=torch.bool)
        read[:, :-1] = mask[:, 1:].bool()
        read[:, :skip_first] = False
        yield ReadBatch(ids.to(device), mask.to(device), read.to(device))


def prompt_batches(
    rows: list[list[int]], positions: list[list[int]], pad_id: int, batch_size: int, device: str | torch.device
) -> Iterator[ReadBatch]:
    for start in range(0, len(rows), batch_size):
        ids, mask = padded(rows[start : start + batch_size], pad_id)
        read = position_mask(positions[start : start + batch_size], ids.shape[1])
        yield ReadBatch(ids.to(device), mask.to(device), read.to(device))


def next_token_log_probs(model: PreTrainedModel, batch: ReadBatch) -> Float[Tensor, "position vocab"]:
    logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False).logits
    return logits[batch.read].float().log_softmax(-1)


def summed_loss(log_probs: Float[Tensor, "position vocab"], targets: Int[Tensor, " position"]) -> float:
    return float(-log_probs.gather(-1, targets[:, None]).sum())


def summed_kl(intact: Float[Tensor, "position vocab"], ablated: Float[Tensor, "position vocab"]) -> float:
    """KL(intact || ablated), summed over positions."""
    return float(F.kl_div(ablated, intact, log_target=True, reduction="sum"))


@torch.no_grad()
def compare(
    model: PreTrainedModel, adapter: ModelAdapter, batches: Iterator[ReadBatch], ablations: list[MeanAblation]
) -> list[Totals]:
    """One intact pass per batch, then one pass per ablation. Losses need targets: read positions must have a real
    next token, else the loss sums are meaningless (as for prompts) and only `kl` counts."""
    totals = [Totals() for _ in ablations]
    for batch in batches:
        intact = next_token_log_probs(model, batch)
        targets = batch.input_ids.roll(-1, dims=1)[batch.read]
        loss_intact = summed_loss(intact, targets)
        for ablation, total in zip(ablations, totals):
            ablation.tokens = batch.attention_mask.bool()
            with ablation.applied(model, adapter):
                ablated = next_token_log_probs(model, batch)
            total.positions += len(targets)
            total.loss_intact += loss_intact
            total.loss_ablated += summed_loss(ablated, targets)
            total.kl += summed_kl(intact, ablated)
            del ablated
    return totals


class PromptKL(BaseModel):
    positions: int
    kl_total: float
    kl_per_position: float


class AblationResult(BaseModel):
    name: str
    layers: list[int]
    ablated_fraction: float
    """Share of (token, neuron) cells ablated over the ablated layers, at real text tokens."""
    ablated_fraction_per_layer: dict[int, float]
    positions: int
    loss_intact: float
    loss_ablated: float
    loss_change: float
    kl_total: float
    kl_per_position: float
    prompts: dict[str, PromptKL] = {}


def ablation_result(name: str, ablation: MeanAblation, fractions: dict[int, float], text: Totals) -> AblationResult:
    return AblationResult(
        name=name,
        layers=ablation.layers,
        ablated_fraction=sum(fractions.values()) / len(fractions),
        ablated_fraction_per_layer=fractions,
        positions=text.positions,
        loss_intact=text.loss_intact / text.positions,
        loss_ablated=text.loss_ablated / text.positions,
        loss_change=(text.loss_ablated - text.loss_intact) / text.positions,
        kl_total=text.kl,
        kl_per_position=text.kl / text.positions,
    )


def prompt_kl(totals: Totals) -> PromptKL:
    return PromptKL(positions=totals.positions, kl_total=totals.kl, kl_per_position=totals.kl / totals.positions)
