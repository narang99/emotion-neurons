"""Every neuron of one MLP: poles, per-band hits and modes, stored for judges and reports.

1. Inputs: one pass over the corpus saves the MLP input at every real token (fp16 memmap under runs/<run_id>/),
   and the layer's gate and up rows.
2. Per chunk of neurons: activations from the saved inputs, `silu(x . gate_n) * (x . up_n)`, then band edges,
   band examples, per-band hits and HDBSCAN modes. The clustering runs in worker processes.
3. Store: one gzipped JSON shard per chunk under runs/<run_id>/store/. See `neuron_entry` for its fields.

- Judges and reports read only the store, so they can run again with new prompts, without the model or a GPU.
- A chunk whose shard exists is skipped, so a stopped run resumes.
- The activations come from the saved inputs, not from the model's own pass: each modes run checks the two agree
  on random tokens (`check_against_model`).
"""

import gc
import gzip
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from safetensors.torch import load_file, save_file
from torch import Tensor

from .config import ModesConfig, ModesRunConfig
from .harvest import CorpusActs, band_edges, band_index, band_names, collect_windows, neuron_examples, pole_hits
from .models import decoder_layers, load_model, load_tokenizer, read_mlp
from .modes import POLES, hit_bands, pole_modes_from_inputs, snippet
from .poles import padded_batch

TOKEN_BLOCK = 65536
"""Tokens per matmul when computing a chunk's activations from the saved inputs."""


def inputs_path(cfg: ModesRunConfig) -> Path:
    return cfg.run_dir / "inputs.f16"


def store_dir(cfg: ModesRunConfig) -> Path:
    return cfg.run_dir / "store"


def save_inputs(cfg: ModesRunConfig) -> None:
    """One pass to the layer: MLP inputs (memmap), token bookkeeping, and the layer's gate and up rows."""
    if (cfg.run_dir / "corpus_meta.json").exists():
        print("[inputs] saved before, skipping the model pass", flush=True)
        return
    t0 = time.time()
    model, tokenizer = load_model(cfg.model, cfg.device)
    windows, sources = collect_windows(cfg.harvest, tokenizer)
    n_tokens = sum(len(w) for w in windows)
    mlp = decoder_layers(model)[cfg.layer].mlp
    d_model = mlp.gate_proj.weight.shape[1]
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(inputs_path(cfg), mode="w+", dtype=np.float16, shape=(n_tokens, d_model))
    no_neurons = torch.zeros(0, dtype=torch.long, device=model.device)
    tokens, window_ids, positions, row = [], [], [], 0
    for start in range(0, len(windows), cfg.harvest.batch_size):
        batch = windows[start : start + cfg.harvest.batch_size]
        ids, mask = padded_batch(batch, tokenizer.pad_token_id)
        mlp_in, _ = read_mlp(model, cfg.layer, ids.to(model.device), mask.to(model.device), no_neurons)
        real = mask.bool()
        x = mlp_in[real.to(mlp_in.device)].half().cpu().numpy()
        X[row : row + len(x)] = x
        row += len(x)
        tokens.append(ids[real])
        window_ids.append((torch.arange(len(batch))[:, None] + start).expand_as(ids)[real])
        pos = torch.arange(ids.shape[1]).expand_as(ids).clone()
        pos[:, : cfg.harvest.skip_first] = -1
        positions.append(pos[real])
        if (start // cfg.harvest.batch_size) % 50 == 0:
            print(f"[inputs] {start + len(batch)}/{len(windows)} windows, {time.time() - t0:.0f}s", flush=True)
    assert row == n_tokens, f"wrote {row} rows, expected {n_tokens}"
    X.flush()
    save_file(
        {"tokens": torch.cat(tokens), "window": torch.cat(window_ids), "position": torch.cat(positions)},
        str(cfg.run_dir / "corpus_tokens.safetensors"),
    )
    gate_up = {"gate": mlp.gate_proj.weight.detach().float().cpu(), "up": mlp.up_proj.weight.detach().float().cpu()}
    save_file(gate_up, str(cfg.run_dir / "gate_up.safetensors"))
    meta = {"n_tokens": n_tokens, "d_model": d_model, "d_mlp": gate_up["gate"].shape[0], "sources": sources}
    (cfg.run_dir / "corpus_meta.json").write_text(json.dumps(meta))
    print(f"[inputs] {n_tokens} tokens x {d_model} saved, {time.time() - t0:.0f}s", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def load_corpus_tokens(cfg: ModesRunConfig) -> tuple[dict, dict[str, Tensor], np.ndarray]:
    meta = json.loads((cfg.run_dir / "corpus_meta.json").read_text())
    t = load_file(str(cfg.run_dir / "corpus_tokens.safetensors"))
    X = np.load(inputs_path(cfg), mmap_mode="r")
    return meta, t, X


def chunk_activations(
    X: np.ndarray, gate: Float[Tensor, "chunk d_model"], up: Float[Tensor, "chunk d_model"], device: str
) -> Float[Tensor, "token chunk"]:
    """fp16 on the CPU: `silu(x . gate_n) * (x . up_n)` for every saved token."""
    g, u = gate.to(device), up.to(device)
    out = torch.empty(len(X), len(gate), dtype=torch.float16)
    for s in range(0, len(X), TOKEN_BLOCK):
        x = torch.from_numpy(np.ascontiguousarray(X[s : s + TOKEN_BLOCK])).to(device).float()
        out[s : s + len(x)] = (torch.nn.functional.silu(x @ g.T) * (x @ u.T)).half().cpu()
    return out


def check_against_model(cfg: ModesRunConfig, corpus: CorpusActs, first: int, count: int = 64) -> float:
    """Largest relative gap between activations from the saved inputs and from a fresh model pass, on random windows.

    `corpus.acts` holds a chunk of neurons starting at neuron `first`.
    """
    model, _ = load_model(cfg.model, cfg.device)
    rng = np.random.default_rng(0)
    local = torch.tensor(rng.choice(corpus.acts.shape[1], 16, replace=False))
    neurons = local + first
    bounds = corpus.window_bounds()
    worst = 0.0
    for w in rng.choice(len(bounds), count, replace=False):
        s, e = bounds[w].tolist()
        ids = corpus.tokens[s:e][None]
        _, a = read_mlp(model, cfg.layer, ids.to(model.device), torch.ones_like(ids).to(model.device), neurons.to(model.device))
        mine = corpus.acts[s:e][:, local].float()
        worst = max(worst, float(((a[0].cpu() - mine).abs() / mine.abs().clamp(min=1.0)).max()))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return worst


def rows_in_order(X: np.ndarray, hits: Int[Tensor, " hit"]) -> Float[np.ndarray, "hit d_model"]:
    """X[hits] in fp32; the memmap is read in ascending order, then put back in the order of `hits`."""
    perm = torch.argsort(hits).numpy()
    Z = np.empty((len(hits), X.shape[1]), dtype=np.float32)
    Z[perm] = X[hits.numpy()[perm]]
    return Z


def cluster_pole_task(args: tuple) -> tuple[int, str, dict]:
    """Worker: one neuron and pole. Inputs arrive as arrays, so no tensors or memmaps cross processes."""
    neuron, pole, hits, Z, a, hb, cfg, seed = args
    return neuron, pole, pole_modes_from_inputs(hits, Z, a, hb, cfg, seed)


def windows_of(tokenizer, corpus: CorpusActs, bounds, items: list[int], acts: Tensor, band, m: ModesConfig) -> list:
    """[[activation, band index, text], ...] for token indices, the firing token marked [[like this]]."""
    return [
        [round(float(acts[t]), 3), int(band[t]), snippet(tokenizer, corpus, bounds, int(t), m.snippet_before, m.snippet_after)]
        for t in items
    ]


def neuron_entry(n: int, j: int, corpus, bounds, tokenizer, edges, examples, modes: dict, cfg) -> dict:
    """One neuron in the store.

    - `edges`: band edges per pole (high ascending from p97, low descending from p3).
    - Per pole: `band_tokens` and `band_hits` (corpus and sample, per band); `clusters` with `bands` (hits per band),
      `pole_pct`, `push_pct`, `mean_act` and `exemplars`; `other` exemplars; `band_examples` and `extreme` windows.
    - Every window is [activation, band index or -1, text].
    """
    a = corpus.acts[:, j].float()
    out = {"neuron": n, "edges": {p: edges[p][:, j].tolist() for p in POLES}, "poles": {}}
    for pole in POLES:
        r = modes[pole]
        band = band_of_tokens(a, corpus.valid, edges[pole][:, j], pole)
        clusters = [
            {**{k: v for k, v in c.items() if k != "exemplars"},
             "exemplars": windows_of(tokenizer, corpus, bounds, c["exemplars"], a, band, cfg.modes)}
            for c in r["clusters"]
        ]  # fmt: skip
        out["poles"][pole] = {
            **{k: r[k] for k in ("n_hits", "band_tokens", "band_hits", "other", "other_bands", "other_pct")},
            "clusters": clusters,
            "other_exemplars": windows_of(tokenizer, corpus, bounds, r["other_exemplars"], a, band, cfg.modes),
            "band_examples": examples[pole],
        }
    return out


def band_of_tokens(a: Tensor, valid, edges: Tensor, pole: str) -> Tensor:
    return band_index(a.masked_fill(~valid, float("nan")), edges, pole)


def run_chunk(cfg: ModesRunConfig, c: int, neurons: list[int], meta, toks, X, gate_up, tokenizer, pool) -> list[dict]:
    t0 = time.time()
    idx = torch.tensor(neurons)
    acts = chunk_activations(X, gate_up["gate"][idx], gate_up["up"][idx], cfg.device)
    corpus = CorpusActs(acts=acts, tokens=toks["tokens"], window=toks["window"], position=toks["position"],
                        sources=[tuple(s) for s in meta["sources"]])  # fmt: skip
    if c == 0:
        worst = check_against_model(cfg, corpus, neurons[0])
        print(f"[check] saved-input activations vs model pass: largest relative gap {worst:.4f}", flush=True)
        assert worst < 0.05, "activations from the saved inputs do not match the model"
    edges = band_edges(corpus, cfg.harvest, torch.device(cfg.device))
    bounds = corpus.window_bounds()
    generator = torch.Generator().manual_seed(cfg.harvest.seed + c)
    examples, tasks = {}, []
    for j, n in enumerate(neurons):
        examples[n] = neuron_examples(corpus, bounds, tokenizer, cfg.harvest, j, edges, generator)
        a = corpus.acts[:, j]
        for pole, hits in pole_hits(corpus, j, edges, cfg.harvest, generator).items():
            hits = hits[: cfg.modes.hits_used]
            hb = hit_bands(a, corpus.valid, edges[pole][:, j], pole, hits)
            Z = rows_in_order(X, hits)
            tasks.append((n, pole, hits, Z, a[hits].float().numpy(), hb, cfg.modes, n))
    print(f"[chunk {c}] activations, edges, hits for {len(neurons)} neurons, {time.time() - t0:.0f}s", flush=True)
    modes: dict[int, dict] = {n: {} for n in neurons}
    for n, pole, r in pool.map(cluster_pole_task, tasks, chunksize=4):
        modes[n][pole] = r
    entries = [neuron_entry(n, j, corpus, bounds, tokenizer, edges, examples[n], modes[n], cfg) for j, n in enumerate(neurons)]
    print(f"[chunk {c}] clustered, {time.time() - t0:.0f}s", flush=True)
    return entries


def run_harvest_all(cfg: ModesRunConfig) -> None:
    """Inputs once, then every chunk of neurons into the store; resumes at the first missing shard."""
    t0 = time.time()
    save_inputs(cfg)
    meta, toks, X = load_corpus_tokens(cfg)
    gate_up = load_file(str(cfg.run_dir / "gate_up.safetensors"))
    tokenizer = load_tokenizer(cfg.model)
    store_dir(cfg).mkdir(parents=True, exist_ok=True)
    chunk = cfg.harvest.neuron_chunk
    starts = list(range(0, meta["d_mlp"], chunk))
    workers = cfg.modes.workers or os.cpu_count()
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    print(f"[harvest-all] {meta['d_mlp']} neurons in {len(starts)} chunks, {workers} workers", flush=True)
    with ProcessPoolExecutor(workers, mp_context=torch.multiprocessing.get_context("spawn")) as pool:
        for c, s in enumerate(starts):
            path = store_dir(cfg) / f"shard_{c:03d}.json.gz"
            if path.exists():
                continue
            entries = run_chunk(cfg, c, list(range(s, min(s + chunk, meta["d_mlp"]))), meta, toks, X, gate_up, tokenizer, pool)
            store = {"bands": {"high": band_names(cfg.harvest.high_quantiles), "low": band_names(cfg.harvest.low_quantiles)},
                     "neurons": entries}  # fmt: skip
            with gzip.open(path.with_suffix(".tmp"), "wt") as f:
                json.dump(store, f)
            path.with_suffix(".tmp").rename(path)
            print(f"[saved] {path.name}: neurons {s}-{s + len(entries) - 1} | {time.time() - t0:.0f}s", flush=True)
    print(f"[harvest-all] done, {len(starts)} shards | {time.time() - t0:.0f}s", flush=True)


def read_store(store: Path) -> dict[int, dict]:
    """{neuron: entry} over every shard, and the band names (key -1)."""
    out: dict[int, dict] = {}
    for path in sorted(store.glob("shard_*.json.gz")):
        with gzip.open(path, "rt") as f:
            s = json.load(f)
        out[-1] = s["bands"]
        out.update({e["neuron"]: e for e in s["neurons"]})
    return out
