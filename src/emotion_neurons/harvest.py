"""Harvest: what chosen neurons fire on, at both poles, over general text.

1. The corpora are cut into windows (as for the poles), and one pass to layer L reads `a_n` for the chosen neurons.
2. Per neuron, band edges at both poles: high p97, p99, p99.9, p99.99 and low p3, p1, p0.1, p0.01.
   - The first `skip_first` tokens of each window are left out of the quantiles, the examples and the hits.
3. Per neuron and pole: example windows in each band, and the most extreme ones.
4. Per neuron and pole: a sample of hit positions (past p97 or p3), for clustering into modes.

- A neuron that happy turns down serves happy through its low pole, so both poles are kept.
"""

import csv
import gc
import json
import time
from pathlib import Path

import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict
from safetensors.torch import load_file, save_file
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .config import HarvestConfig, ModesRunConfig
from .models import load_model, read_mlp
from .poles import padded_batch, quantiles_per_neuron
from .texts import corpus_windows


class CorpusActs(BaseModel):
    """Every real token of the corpus, in window order."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    acts: Float[Tensor, "token neuron"]  # fp16, CPU
    tokens: Int[Tensor, " token"]
    window: Int[Tensor, " token"]  # index into `sources`
    position: Int[Tensor, " token"]  # position in its window
    sources: list[tuple[str, int]]  # (corpus, document id) per window

    @property
    def valid(self) -> Bool[Tensor, " token"]:
        return self.position >= 0

    def window_bounds(self) -> Int[Tensor, "window 2"]:
        """[start, end) token index of each window; tokens are stored window after window."""
        counts = torch.bincount(self.window, minlength=len(self.sources))
        ends = counts.cumsum(0)
        return torch.stack([ends - counts, ends], 1)


def save_corpus(corpus: CorpusActs, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    tensors = {"acts": corpus.acts.contiguous(), "tokens": corpus.tokens, "window": corpus.window, "position": corpus.position}
    save_file(tensors, str(run_dir / "corpus.safetensors"))
    (run_dir / "sources.json").write_text(json.dumps(corpus.sources))


def load_corpus(run_dir: Path) -> CorpusActs:
    t = load_file(str(run_dir / "corpus.safetensors"))
    sources = [tuple(s) for s in json.loads((run_dir / "sources.json").read_text())]
    return CorpusActs(acts=t["acts"], tokens=t["tokens"], window=t["window"], position=t["position"], sources=sources)


def neurons_until(csv_path: Path, rebuilt_pct: float) -> list[int]:
    """The neurons of a greedy-set CSV, in rank order, up to the one that reaches `rebuilt_pct` of the write."""
    neurons = []
    for row in csv.DictReader(csv_path.open()):
        neurons.append(int(row["neuron"]))
        if float(row["rebuilt_after_pct"]) >= rebuilt_pct:
            return neurons
    raise ValueError(f"{csv_path} never reaches {rebuilt_pct}% (its set stops at {row['rebuilt_after_pct']}%)")


def control_neurons(exclude: list[int], n_neurons: int, count: int, seed: int) -> list[int]:
    rest = torch.tensor(sorted(set(range(n_neurons)) - set(exclude)))
    pick = torch.randperm(len(rest), generator=torch.Generator().manual_seed(seed))[:count]
    return sorted(rest[pick].tolist())


def collect_windows(cfg: HarvestConfig, tokenizer: PreTrainedTokenizerBase) -> tuple[list[list[int]], list[tuple[str, int]]]:
    windows, sources = [], []
    for corpus in cfg.corpora:
        start = len(windows)
        for doc_id, w in corpus_windows(corpus, tokenizer, cfg.window, cfg.min_window, cfg.seed):
            windows.append(w)
            sources.append((corpus.name, doc_id))
        print(
            f"[corpus] {corpus.name}: {len(windows) - start} windows, {sum(len(w) for w in windows[start:])} tokens", flush=True
        )
    return windows, sources


def read_corpus(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, cfg: HarvestConfig, layer: int, neurons: list[int]
) -> CorpusActs:
    """One pass to layer L. Tokens before `skip_first` get position -1, so `valid` leaves them out."""
    windows, sources = collect_windows(cfg, tokenizer)
    index = torch.tensor(neurons, device=model.device)
    acts, tokens, window_ids, positions = [], [], [], []
    for start in range(0, len(windows), cfg.batch_size):
        batch = windows[start : start + cfg.batch_size]
        ids, mask = padded_batch(batch, tokenizer.pad_token_id)
        _, a = read_mlp(model, layer, ids.to(model.device), mask.to(model.device), index)
        real = mask.bool()
        acts.append(a[real.to(a.device)].half().cpu())
        tokens.append(ids[real])
        window_ids.append((torch.arange(len(batch))[:, None] + start).expand_as(ids)[real])
        pos = torch.arange(ids.shape[1]).expand_as(ids).clone()
        pos[:, : cfg.skip_first] = -1
        positions.append(pos[real])
        if (start // cfg.batch_size) % 100 == 0:
            print(f"[harvest] {start + len(batch)}/{len(windows)} windows", flush=True)
    return CorpusActs(
        acts=torch.cat(acts), tokens=torch.cat(tokens), window=torch.cat(window_ids),
        position=torch.cat(positions), sources=sources,
    )  # fmt: skip


def band_edges(corpus: CorpusActs, cfg: HarvestConfig, device: torch.device) -> dict[str, Float[Tensor, "edge neuron"]]:
    """`high`: values at high_quantiles, ascending. `low`: values at low_quantiles, descending (p3 first)."""
    valid = corpus.acts[corpus.valid]
    return {
        "high": quantiles_per_neuron(valid, cfg.high_quantiles, device),
        "low": quantiles_per_neuron(valid, cfg.low_quantiles, device),
    }


def band_masks(a: Float[Tensor, " token"], edges: Float[Tensor, " edge"], pole: str) -> list[Bool[Tensor, " token"]]:
    """One mask per band between consecutive edges; the last band is open-ended (past the most extreme edge)."""
    outer = [*edges.tolist(), float("inf") if pole == "high" else -float("inf")]
    if pole == "high":
        return [(a >= lo) & (a < hi) for lo, hi in zip(outer[:-1], outer[1:])]
    return [(a <= hi) & (a > lo) for hi, lo in zip(outer[:-1], outer[1:])]


def sample(mask: Bool[Tensor, " token"], count: int, generator: torch.Generator) -> Int[Tensor, " pick"]:
    hits = mask.nonzero().squeeze(1)
    return hits[torch.randperm(len(hits), generator=generator)[:count]]


def window_text(
    tokenizer: PreTrainedTokenizerBase, corpus: CorpusActs, bounds: Int[Tensor, "window 2"], i: int, before: int, after: int
) -> str:
    """The token at i marked as [[token]], with up to `before` tokens before and `after` after, inside its window."""
    start, end = bounds[corpus.window[i]].tolist()
    lo, hi = max(start, i - before), min(end, i + after + 1)
    ids = corpus.tokens
    return tokenizer.decode(ids[lo:i]) + "[[" + tokenizer.decode(ids[i : i + 1]) + "]]" + tokenizer.decode(ids[i + 1 : hi])


def neuron_examples(
    corpus: CorpusActs,
    bounds: Int[Tensor, "window 2"],
    tokenizer: PreTrainedTokenizerBase,
    cfg: HarvestConfig,
    j: int,
    edges: dict,
    generator: torch.Generator,
) -> dict:
    """{pole: {band name: [[activation, text], ...], "extreme": [...]}} for column j."""
    a = corpus.acts[:, j].float().masked_fill(~corpus.valid, float("nan"))

    def shown(picks: Tensor) -> list:
        return [[round(float(a[i]), 3), window_text(tokenizer, corpus, bounds, int(i), cfg.before, cfg.after)] for i in picks]

    out = {}
    for pole, quantiles in (("high", cfg.high_quantiles), ("low", cfg.low_quantiles)):
        pole_out = {
            name: shown(sample(mask, cfg.examples_per_band, generator))
            for name, mask in zip(band_names(quantiles), band_masks(a, edges[pole][:, j], pole))
        }
        extreme = a.nan_to_num(-float("inf") if pole == "high" else float("inf"))
        pole_out["extreme"] = shown(extreme.topk(cfg.extreme_examples, largest=pole == "high").indices)
        out[pole] = pole_out
    return out


def band_names(quantiles: list[float]) -> list[str]:
    """e.g. high [0.97, 0.99, 0.999, 0.9999] -> p97-p99, p99-p99.9, p99.9-p99.99, p99.99+."""
    labels = [f"p{100 * q:g}" for q in quantiles]
    return [f"{a}-{b}" for a, b in zip(labels[:-1], labels[1:])] + [f"{labels[-1]}{'+' if quantiles[0] > 0.5 else '-'}"]


def pole_hits(corpus: CorpusActs, j: int, edges: dict, count: int, generator) -> dict[str, Int[Tensor, " hit"]]:
    """A sample of token indices past p97 (high) and past p3 (low), for the mode clustering."""
    a = corpus.acts[:, j].float()
    return {
        "high": sample(corpus.valid & (a >= edges["high"][0, j]), count, generator),
        "low": sample(corpus.valid & (a <= edges["low"][0, j]), count, generator),
    }


def corpus_for(cfg: ModesRunConfig, neurons: list[int]) -> CorpusActs:
    """The saved corpus pass if an earlier run got that far, else one pass to layer L, saved at once.

    The model is freed before returning: sorting activations for the band edges needs the GPU memory.
    """
    if (cfg.run_dir / "corpus.safetensors").exists():
        print(f"[corpus] loading the saved pass from {cfg.run_dir}", flush=True)
        return load_corpus(cfg.run_dir)
    model, tokenizer = load_model(cfg.model, cfg.device)
    corpus = read_corpus(model, tokenizer, cfg.harvest, cfg.layer, neurons)
    save_corpus(corpus, cfg.run_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return corpus


def run_harvest(cfg: ModesRunConfig) -> None:
    """The harvest stage: writes results/<run_id>/harvest.json and runs/<run_id>/{corpus,hits}.safetensors, sources.json."""
    h = cfg.harvest
    t0 = time.time()
    chosen = neurons_until(h.neurons_csv, h.rebuilt_pct)
    controls = control_neurons(chosen, AutoConfig.from_pretrained(cfg.model).intermediate_size, h.n_controls, h.seed)
    neurons = chosen + controls
    print(f"[neurons] {len(chosen)} from {h.neurons_csv} (to {h.rebuilt_pct}%) + {len(controls)} controls", flush=True)

    corpus = corpus_for(cfg, neurons)
    assert corpus.acts.shape[1] == len(neurons), f"saved corpus has {corpus.acts.shape[1]} neurons, expected {len(neurons)}"
    print(f"[corpus] {len(corpus.tokens)} tokens, {int(corpus.valid.sum())} valid, {time.time() - t0:.0f}s", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    edges = band_edges(corpus, h, torch.device(cfg.device))
    print(f"[edges] both poles, {time.time() - t0:.0f}s", flush=True)
    bounds = corpus.window_bounds()
    generator = torch.Generator().manual_seed(h.seed)
    examples, hits = {}, {}
    for j, n in enumerate(neurons):
        examples[n] = neuron_examples(corpus, bounds, tokenizer, h, j, edges, generator)
        for pole, idx in pole_hits(corpus, j, edges, h.hits_per_pole, generator).items():
            hits[f"{n}/{pole}"] = idx.contiguous()
    print(f"[examples] {len(neurons)} neurons, {time.time() - t0:.0f}s", flush=True)
    save_file(hits, str(cfg.run_dir / "hits.safetensors"))

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    harvest = {
        "model": cfg.model, "layer": cfg.layer, "neurons": chosen, "controls": controls,
        "tokens": len(corpus.tokens), "valid_tokens": int(corpus.valid.sum()), "config": h.model_dump(mode="json"),
        "bands": {"high": band_names(h.high_quantiles), "low": band_names(h.low_quantiles)},
        "edges": {str(n): {pole: e[:, j].tolist() for pole, e in edges.items()} for j, n in enumerate(neurons)},
        "examples": {str(n): v for n, v in examples.items()},
    }  # fmt: skip
    (cfg.result_dir / "harvest.json").write_text(json.dumps(harvest, ensure_ascii=False))
    print(f"[saved] {cfg.result_dir}/harvest.json and {cfg.run_dir}/ | {time.time() - t0:.0f}s", flush=True)
