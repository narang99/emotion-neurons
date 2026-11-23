"""Modes: clusters of what a neuron reads when it fires at one pole (from tag exp/stage3_l25, now for both poles).

1. The MLP input (post_attention_layernorm output) at every sampled hit token, from one pass over the windows
   that hold them.
2. Per neuron and pole: L2-normalise the inputs, PCA, HDBSCAN.
   - A noise hit joins the nearest cluster when its cosine to that centre is above the cluster's own
     `assign_quantile` of member cosines; the rest stay "other".
3. Per cluster: the exemplars closest to its centre, for Haiku to name.
4. Per cluster: how many of its hits fall in each band, and its weighted share of the pole.
   - A hit weighs (tokens of its band in the corpus) / (hits of its band in the sample). With a per-band sample
     this undoes the oversampling of the extreme bands, so the shares still describe the whole pole.
   - `pole_pct`: the cluster's estimated share of the pole's tokens. `push_pct`: its share of the pole's summed
     |activation|, i.e. of what the pole writes.
"""

import json
import time
from collections import defaultdict

import numpy as np
import torch
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict
from safetensors.torch import load_file
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import ModesConfig, ModesRunConfig
from .harvest import CorpusActs, band_index, load_corpus
from .models import load_model, read_mlp
from .poles import padded_batch

POLES = ("high", "low")


def inputs_at(
    model: PreTrainedModel, corpus: CorpusActs, layer: int, hit_tokens: Int[Tensor, " hit"], batch_size: int
) -> dict[int, Float[Tensor, " d_model"]]:
    """{token index: MLP input}, fp16 on the CPU, re-run from the windows that hold the hits."""
    bounds = corpus.window_bounds()
    by_window = defaultdict(list)
    for t in hit_tokens.tolist():
        by_window[int(corpus.window[t])].append(t)
    windows = sorted(by_window)
    no_neurons = torch.zeros(0, dtype=torch.long, device=model.device)
    out = {}
    for start in range(0, len(windows), batch_size):
        batch = windows[start : start + batch_size]
        seqs = [corpus.tokens[bounds[w, 0] : bounds[w, 1]].tolist() for w in batch]
        ids, mask = padded_batch(seqs, pad_id=0)
        mlp_in, _ = read_mlp(model, layer, ids.to(model.device), mask.to(model.device), no_neurons)
        for j, w in enumerate(batch):
            for t in by_window[w]:
                out[t] = mlp_in[j, t - int(bounds[w, 0])].half().cpu()
        if (start // batch_size) % 200 == 0:
            print(f"[inputs] {start + len(batch)}/{len(windows)} windows", flush=True)
    return out


def recomputed_activation(model: PreTrainedModel, layer: int, neuron: int, x: Float[Tensor, " d_model"]) -> float:
    mlp = model.model.layers[layer].mlp
    x = x.float().to(model.device)
    return float(torch.nn.functional.silu(mlp.gate_proj.weight[neuron].float() @ x) * (mlp.up_proj.weight[neuron].float() @ x))


def cluster_hits(Z: np.ndarray, cfg: ModesConfig) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """HDBSCAN labels, labels after assigning close noise points, and the unit centre of each cluster."""
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    if len(Z) < 2 * cfg.min_cluster_size:
        labels = np.full(len(Z), -1)
    else:
        P = PCA(n_components=min(cfg.pca_dims, len(Z) - 1, Z.shape[1]), random_state=0).fit_transform(Z)
        labels = HDBSCAN(min_cluster_size=cfg.min_cluster_size, min_samples=cfg.min_samples, copy=True).fit_predict(P)
    centres = {}
    for c in sorted(set(labels) - {-1}):
        centre = Z[labels == c].mean(0)
        centres[c] = centre / np.linalg.norm(centre)
    final = labels.copy()
    if centres:
        keys = list(centres)
        cos = Z @ np.stack([centres[c] for c in keys]).T
        cut = {c: np.quantile(cos[labels == c, i], cfg.assign_quantile) for i, c in enumerate(keys)}
        for i in np.nonzero(labels == -1)[0]:
            best = int(cos[i].argmax())
            if cos[i, best] >= cut[keys[best]]:
                final[i] = keys[best]
    return labels, final, {c: v for c, v in centres.items()}


class HitBands(BaseModel):
    """The band of each sampled hit, its weight, and the pole's band sizes in the corpus and in the sample."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    band: Int[np.ndarray, " hit"]
    weight: Float[np.ndarray, " hit"]
    tokens: list[int]  # per band, in the corpus
    sampled: list[int]  # per band, among the hits


def hit_bands(a: Float[Tensor, " token"], valid, edges: Float[Tensor, " edge"], pole: str, hits) -> HitBands:
    bands = band_index(a.float().masked_fill(~valid, float("nan")), edges, pole)
    n_bands = len(edges)
    tokens = torch.bincount(bands[bands >= 0], minlength=n_bands)
    band = bands[hits]
    assert (band >= 0).all(), "a hit lies outside its pole"
    sampled = torch.bincount(band, minlength=n_bands)
    weight = tokens[band].double() / sampled[band].double()
    return HitBands(band=band.numpy(), weight=weight.numpy(), tokens=tokens.tolist(), sampled=sampled.tolist())


def pole_modes(
    hits: Int[Tensor, " hit"], X: dict[int, Tensor], a: Float[np.ndarray, " hit"], hb: HitBands, cfg: ModesConfig, seed: int
) -> dict:
    """Clusters of one neuron's hits at one pole: size, band mix, weighted shares, exemplar token indices."""
    Z = torch.stack([X[t] for t in hits.tolist()]).float().numpy()
    labels, final, centres = cluster_hits(Z, cfg)
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    w, push = hb.weight, hb.weight * np.abs(a)
    clusters = []
    for c, centre in centres.items():
        members = np.nonzero(labels == c)[0]
        closest = members[np.argsort(-(Zn[members] @ centre))[: cfg.exemplars]]
        mine = final == c
        clusters.append({
            "id": int(c), "size": int(len(members)), "with_assigned": int(mine.sum()),
            "mean_act": float((a[mine] * w[mine]).sum() / w[mine].sum()),
            "bands": np.bincount(hb.band[mine], minlength=len(hb.tokens)).tolist(),
            "pole_pct": float(100 * w[mine].sum() / w.sum()), "push_pct": float(100 * push[mine].sum() / push.sum()),
            "exemplars": [int(hits[i]) for i in closest],
        })  # fmt: skip
    noise = np.nonzero(final == -1)[0]
    rng = np.random.default_rng(seed)
    other = rng.choice(noise, min(cfg.other_exemplars, len(noise)), replace=False) if len(noise) else []
    return {
        "n_hits": int(len(hits)),
        "band_tokens": hb.tokens,
        "band_hits": hb.sampled,
        "clusters": clusters,
        "other": int(len(noise)),
        "other_bands": np.bincount(hb.band[noise], minlength=len(hb.tokens)).tolist(),
        "other_pct": float(100 * w[noise].sum() / w.sum()),
        "other_exemplars": [int(hits[i]) for i in other],
    }


def check_inputs(model, layer: int, corpus: CorpusActs, X: dict, hits: dict, column: dict, count: int, seed: int) -> float:
    """Largest relative gap between recomputed and harvested activations over `count` random hits."""
    rng = np.random.default_rng(seed)
    keys = list(hits)
    worst = 0.0
    for k in rng.choice(len(keys), min(count, len(keys)), replace=False):
        neuron = int(keys[k].split("/")[0])
        t = int(hits[keys[k]][0])
        harvested = float(corpus.acts[t, column[neuron]])
        recomputed = recomputed_activation(model, layer, neuron, X[t])
        worst = max(worst, abs(recomputed - harvested) / max(1.0, abs(harvested)))
    return worst


def snippet(tokenizer: PreTrainedTokenizerBase, corpus: CorpusActs, bounds, t: int, before: int, after: int) -> str:
    start, end = bounds[corpus.window[t]].tolist()
    ids = corpus.tokens
    lo, hi = max(start, t - before), min(end, t + after + 1)
    text = tokenizer.decode(ids[lo:t]) + "[[" + tokenizer.decode(ids[t : t + 1]) + "]]" + tokenizer.decode(ids[t + 1 : hi])
    return text.replace("\n", " ")


def batch_groups(result: dict, neurons: list[int], m: ModesConfig) -> list[list[int]]:
    """Neurons per batch file: at most `neurons_per_batch`, and at most `clusters_per_batch` clusters.

    - A judge answers one line per cluster, so the cluster count bounds the reply; past ~150 lines it gets cut off.
    - A neuron with more clusters than the budget gets a file of its own.
    """
    groups, current, clusters = [], [], 0
    for n in neurons:
        size = sum(len(result[n][p]["clusters"]) for p in POLES)
        if current and (len(current) == m.neurons_per_batch or clusters + size > m.clusters_per_batch):
            groups.append(current)
            current, clusters = [], 0
        current.append(n)
        clusters += size
    return groups + ([current] if current else [])


def batch_files(result: dict, neurons: list[int], corpus: CorpusActs, column: dict, tokenizer, m: ModesConfig) -> list[str]:
    """Text for the judges: per neuron and pole, each cluster's exemplars with their activation."""
    bounds = corpus.window_bounds()
    files = []
    for group in batch_groups(result, neurons, m):
        lines = []
        for n in group:
            lines.append(f"##### neuron {n}")
            for pole in POLES:
                r = result[n][pole]
                lines.append(f"=== {pole} pole ({'p97 and above' if pole == 'high' else 'p3 and below'})")
                groups = [(f"cluster {c['id']}", c["exemplars"]) for c in r["clusters"]]
                groups.append(("unclustered (sample)", r["other_exemplars"]))
                for name, exemplars in groups:
                    if not exemplars:
                        continue
                    lines.append(f"--- {name}")
                    for t in exemplars:
                        a = float(corpus.acts[t, column[n]])
                        lines.append(f"  [{a:.2f}] {snippet(tokenizer, corpus, bounds, t, m.snippet_before, m.snippet_after)}")
        files.append("\n".join(lines) + "\n")
    return files


def run_modes(cfg: ModesRunConfig) -> None:
    """The modes stage: writes results/<run_id>/modes.json and results/<run_id>/modes/batch_XX.txt."""
    m = cfg.modes
    t0 = time.time()
    harvest = json.loads((cfg.result_dir / "harvest.json").read_text())
    neurons = harvest["neurons"] + harvest["controls"]
    column = {n: j for j, n in enumerate(neurons)}
    corpus = load_corpus(cfg.run_dir)
    hits = {k: v[: m.hits_used] for k, v in load_file(str(cfg.run_dir / "hits.safetensors")).items()}
    needed = torch.unique(torch.cat(list(hits.values())))
    print(f"[hits] {len(neurons)} neurons x 2 poles, {len(needed)} distinct hit tokens, {time.time() - t0:.0f}s", flush=True)

    model, tokenizer = load_model(cfg.model, cfg.device)
    X = inputs_at(model, corpus, cfg.layer, needed, m.batch_size)
    worst = check_inputs(model, cfg.layer, corpus, X, hits, column, m.checks, seed=0)
    print(f"[check] recomputed vs harvested activation: largest relative gap {worst:.3f} over {m.checks} hits", flush=True)
    assert worst < 0.05, "recomputed activations do not match the harvest"

    result = {}
    for i, n in enumerate(neurons, 1):
        result[n] = {}
        a = corpus.acts[:, column[n]]
        for pole in POLES:
            idx = hits[f"{n}/{pole}"]
            hb = hit_bands(a, corpus.valid, torch.tensor(harvest["edges"][str(n)][pole]), pole, idx)
            result[n][pole] = pole_modes(idx, X, a[idx].float().numpy(), hb, m, seed=n)
        if i % 50 == 0 or i == len(neurons):
            print(f"[cluster] {i}/{len(neurons)} neurons, {time.time() - t0:.0f}s", flush=True)
    sizes = [len(result[n][p]["clusters"]) for n in neurons for p in POLES]
    print(f"[clusters] median {np.median(sizes):.0f} per neuron and pole, {sum(s == 0 for s in sizes)} with none", flush=True)

    (cfg.result_dir / "modes.json").write_text(json.dumps({str(n): v for n, v in result.items()}))
    out = cfg.result_dir / "modes"
    out.mkdir(parents=True, exist_ok=True)
    files = batch_files(result, neurons, corpus, column, tokenizer, m)
    for b, text in enumerate(files):
        (out / f"batch_{b:02d}.txt").write_text(text)
    print(
        f"[saved] modes.json and {len(files)} batch files, ~{sum(len(f) for f in files) // 4} tokens | {time.time() - t0:.0f}s",
        flush=True,
    )
