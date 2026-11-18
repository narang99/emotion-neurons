"""Activations of chosen neurons over a mixed corpus, and their percentile bands.

1. Each corpus is streamed, tokenized and cut into windows (never across documents).
2. One pass to layer L writes `a_n = silu(gate_n) * up_n` for the chosen neurons to a memmap [token, neuron].
3. Per neuron, over real tokens past `skip_first`:
   - the percentile thresholds (p97, p99, p99.9, p99.99);
   - examples per band, with 60 tokens before and 8 after;
   - a sample of p97+ hit indices for the mode clustering.

- The first tokens of each window are left out, because attention-sink effects dominate there.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from numpy.typing import NDArray
from pydantic import BaseModel
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import HarvestConfig, RunConfig
from .decomposition.dims_then_neurons import rank_neurons
from .models import read_mlp
from .texts import corpus_windows
from .vectors import EmotionVectors


class Windows(BaseModel):
    """Token windows in index order; `corpus_doc[i]` is window i's (corpus name, document id)."""

    tokens: list[list[int]]
    corpus_doc: list[tuple[str, int]]

    @property
    def offsets(self) -> NDArray[np.int64]:
        out = np.zeros(len(self.tokens) + 1, dtype=np.int64)
        out[1:] = np.cumsum([len(w) for w in self.tokens])
        return out


def collect_windows(h: HarvestConfig, tokenizer: PreTrainedTokenizerBase) -> Windows:
    tokens, corpus_doc = [], []
    for corpus in h.corpora:
        start = len(tokens)
        for doc_id, window in corpus_windows(corpus, tokenizer, h.window, h.min_window, h.seed):
            tokens.append(window)
            corpus_doc.append((corpus.name, doc_id))
        print(f"[corpus] {corpus.name}: {len(tokens) - start} windows, {sum(len(w) for w in tokens[start:])} tokens")
    return Windows(tokens=tokens, corpus_doc=corpus_doc)


class NeuronChoice(BaseModel):
    chosen: list[int]
    controls: list[int]
    groups: dict[str, list[int]]


def choose_neurons(cfg: RunConfig, W_down: Float[torch.Tensor, "d_model d_mlp"]) -> NeuronChoice:
    """The neurons of `harvest.neurons_file` if given; else the Stage 3 set if small enough, else its ranked top.

    - Plus random controls from the rest.
    """
    h = cfg.harvest
    if h.neurons_file:
        selection = json.loads(h.neurons_file.read_text())
        groups, chosen = selection["groups"], sorted(selection["neurons"])
        print(f"[neurons] {len(chosen)} from {h.neurons_file}")
    else:
        neuron_set = json.loads((cfg.result_dir / "neuron_set.json").read_text())
        union = sorted({n for ids in neuron_set["members"].values() for n in ids})
        if len(union) <= h.max_neurons:
            groups, chosen = {"set": union}, union
        else:
            vectors = EmotionVectors.load(cfg.results_dir / cfg.neuron_set.vectors_run / "vectors.safetensors")
            groups = rank_neurons(neuron_set, W_down, vectors.get("mlp_out"), vectors.get("mlp_act"), h.per_group)
            chosen = sorted({n for g in groups.values() for n in g})
            print(
                f"[neurons] Stage 3 set is {len(union)} neurons (> {h.max_neurons}); taking the top {h.per_group} per (emotion, pole)"
            )
    rest = sorted(set(range(W_down.shape[1])) - set(chosen))
    controls = sorted(random.Random(h.seed).sample(rest, min(h.n_controls, len(rest))))
    return NeuronChoice(chosen=chosen, controls=controls, groups=groups)


def padded_batch(windows: list[list[int]], pad_id: int) -> tuple[Int[torch.Tensor, "batch seq"], Int[torch.Tensor, "batch seq"]]:
    longest = max(len(w) for w in windows)
    ids = torch.full((len(windows), longest), pad_id, dtype=torch.long)
    mask = torch.zeros((len(windows), longest), dtype=torch.long)
    for j, w in enumerate(windows):
        ids[j, : len(w)] = torch.tensor(w)
        mask[j, : len(w)] = 1
    return ids, mask


class TokenArrays(BaseModel):
    """Per corpus token: its id, position in its window, and window index."""

    model_config = {"arbitrary_types_allowed": True}

    tokens: NDArray[np.int32]
    pos: NDArray[np.int16]
    window: NDArray[np.int32]


def activation_pass(
    model: PreTrainedModel, layer: int, windows: Windows, neurons: list[int], batch_size: int, pad_id: int, acts_path: Path
) -> TokenArrays:
    """Writes the memmap at `acts_path`; similar-length windows share a batch."""
    offsets = windows.offsets
    total = int(offsets[-1])
    acts = np.lib.format.open_memmap(acts_path, mode="w+", dtype=np.float16, shape=(total, len(neurons)))
    arrays = TokenArrays(tokens=np.zeros(total, np.int32), pos=np.zeros(total, np.int16), window=np.zeros(total, np.int32))
    idx = torch.tensor(neurons, device=model.device)
    order = sorted(range(len(windows.tokens)), key=lambda i: -len(windows.tokens[i]))
    for start in tqdm(range(0, len(order), batch_size), desc="harvest"):
        batch = order[start : start + batch_size]
        ids, mask = padded_batch([windows.tokens[i] for i in batch], pad_id)
        _, a = read_mlp(model, layer, ids.to(model.device), mask.to(model.device), idx)
        assert torch.isfinite(a).all()
        a = a.half().cpu().numpy()
        for j, i in enumerate(batch):
            n, o = len(windows.tokens[i]), offsets[i]
            acts[o : o + n] = a[j, :n]
            arrays.tokens[o : o + n] = windows.tokens[i]
            arrays.pos[o : o + n] = np.arange(n)
            arrays.window[o : o + n] = i
    acts.flush()
    return arrays


class Snippet(BaseModel):
    corpus: str
    before: str
    token: str
    after: str


def snippet(t: int, windows: Windows, arrays: TokenArrays, tokenizer: PreTrainedTokenizerBase) -> Snippet:
    """60 tokens before and 8 after token t, within its window."""
    w = arrays.window[t]
    offsets = windows.offsets
    lo, hi = max(offsets[w], t - 60), min(offsets[w + 1], t + 9)
    return Snippet(
        corpus=windows.corpus_doc[w][0],
        before=tokenizer.decode(arrays.tokens[lo:t].tolist()),
        token=tokenizer.decode([int(arrays.tokens[t])]),
        after=tokenizer.decode(arrays.tokens[t + 1 : hi].tolist()),
    )


def percentile_thresholds(acts: np.memmap, valid: NDArray[np.bool_], h: HarvestConfig, device: str) -> NDArray[np.float32]:
    """[neuron, percentile], by k-th value on the GPU, a chunk of neurons at a time."""
    n_valid = int(valid.sum())
    thresholds = np.zeros((acts.shape[1], len(h.percentiles)), dtype=np.float32)
    valid_t = torch.from_numpy(valid)
    for c0 in tqdm(range(0, acts.shape[1], h.percentile_chunk), desc="percentiles"):
        cols = torch.from_numpy(np.asarray(acts[:, c0 : c0 + h.percentile_chunk], dtype=np.float32))[valid_t].to(device)
        for k, q in enumerate(h.percentiles):
            kth = max(1, int(round(q * n_valid)))
            thresholds[c0 : c0 + cols.shape[1], k] = cols.kthvalue(kth, dim=0).values.cpu().numpy()
    return thresholds


def neuron_bands(
    values: NDArray[np.float32], thresholds: NDArray[np.float32], valid_idx: NDArray[np.int64], h: HarvestConfig,
    rng: np.random.Generator, describe,
) -> tuple[dict, NDArray[np.int64], NDArray[np.int64]]:  # fmt: skip
    """Examples per band, the sampled p97+ hits (global token indices), and all p97+ hits (valid indices).

    - `values` are one neuron's activations on the valid tokens; `describe(i)` turns a valid index into a snippet dict.
    """
    above = np.nonzero(values >= thresholds[0])[0]
    take = above if len(above) <= h.hits_per_neuron else rng.choice(above, h.hits_per_neuron, replace=False)
    top = above[np.argsort(-values[above])[: h.top_k]]
    bands = {}
    edges = list(thresholds) + [np.inf]
    for b in range(len(h.percentiles)):
        inside = np.nonzero((values >= edges[b]) & (values < edges[b + 1]))[0]
        pick = inside if len(inside) <= h.examples_per_band else rng.choice(inside, h.examples_per_band, replace=False)
        bands[f"p{h.percentiles[b] * 100:g}"] = [{"act": float(values[i]), **describe(i)} for i in pick]
    bands["top"] = [{"act": float(values[i]), **describe(i)} for i in top[: h.examples_per_band]]
    return bands, valid_idx[np.union1d(take, top)], above
