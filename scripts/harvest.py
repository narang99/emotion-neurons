"""Stage 4: activations of the Stage 3 neurons (plus random controls) over a mixed text corpus. GPU.

    python scripts/harvest.py --config configs/stage3_l25.yaml

1. Stream each corpus, tokenize, cut each document into windows of `window` tokens (never across documents).
2. Run the model up to layer L; write a_n = silu(gate_n) * up_n for the chosen neurons to a memmap.
3. Per neuron: p97, p99, p99.9, p99.99 of a_n over real tokens (the first `skip_first` of each window are excluded,
   where attention-sink effects dominate); examples per band with 60 tokens before and 8 after; a sample of p97+ hit
   indices for the mode clustering (scripts/modes.py).

runs/<run_id>/: acts.f16 [T, N], tokens.i32 [T], pos.i16 [T], window.i32 [T], windows.json (corpus, doc id per window)
results/<run_id>/: harvest.json (neurons, thresholds, examples), hits.safetensors (sampled hit indices per neuron)
"""

import json
import random
import sys
import time

import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from emotion_neurons.config import load_run_config
from emotion_neurons.model import StopForward, capture, load_model
from emotion_neurons.selection import rank_neurons


def doc_text(row, field):
    v = row[field]
    if isinstance(v, list):  # chat: list of {role, content}
        return "\n\n".join(f"{m['role']}: {m['content']}" for m in v)
    return v


def windows_for(corpus, tok, window, min_len, seed):
    ds = load_dataset(corpus.path, corpus.config, split=corpus.split, streaming=True).shuffle(seed=seed, buffer_size=1_000)
    total = 0
    for doc_id, row in enumerate(ds):
        ids = tok(doc_text(row, corpus.field), add_special_tokens=False)["input_ids"]
        for s in range(0, len(ids), window):
            w = ids[s : s + window]
            if len(w) < min_len:
                break
            yield doc_id, w
            total += len(w)
            if total >= corpus.tokens:
                return


def choose_neurons(cfg, model):
    """The whole Stage 3 set if it is small enough, else the top `per_group` per (emotion, pole); plus controls."""
    ns = json.loads((cfg.result_dir / "neuron_set.json").read_text())
    W = model.model.layers[cfg.layer].mlp.down_proj.weight.float().cpu()
    union = sorted({n for ids in ns["members"].values() for n in ids})
    if len(union) <= cfg.harvest.max_neurons:
        groups, chosen = {"set": union}, union
    else:
        vec = load_file(str(cfg.results_dir / cfg.neuron_set.vectors_run / "vectors.safetensors"))
        groups = rank_neurons(ns, W, vec, cfg.harvest.per_group)
        chosen = sorted({n for g in groups.values() for n in g})
        print(f"[neurons] Stage 3 set is {len(union)} neurons (> {cfg.harvest.max_neurons}); "
              f"taking the top {cfg.harvest.per_group} per (emotion, pole)")
    rng = random.Random(cfg.harvest.seed)
    rest = sorted(set(range(W.shape[1])) - set(chosen))
    controls = sorted(rng.sample(rest, min(cfg.harvest.n_controls, len(rest))))
    return chosen, controls, groups


def main():
    cfg = load_run_config(sys.argv[1:])
    h = cfg.harvest
    t0 = time.time()
    model, tok = load_model(cfg.model, cfg.device)
    chosen, controls, groups = choose_neurons(cfg, model)
    neurons = chosen + controls
    idx = torch.tensor(neurons, device=cfg.device)
    N = len(neurons)
    print(f"[neurons] {len(chosen)} from the set + {len(controls)} controls = {N}")

    # Windows first (CPU), so the memmaps have their final size.
    wins, meta = [], []
    for c in h.corpora:
        n0 = len(wins)
        for doc_id, w in windows_for(c, tok, h.window, h.min_window, h.seed):
            wins.append(w)
            meta.append([c.name, doc_id])
        print(f"[corpus] {c.name}: {len(wins) - n0} windows, {sum(len(w) for w in wins[n0:])} tokens")
    T = sum(len(w) for w in wins)
    print(f"[corpus] total {T} tokens in {len(wins)} windows, {time.time() - t0:.0f}s")

    rd = cfg.run_dir
    rd.mkdir(parents=True, exist_ok=True)
    acts = np.lib.format.open_memmap(rd / "acts.npy", mode="w+", dtype=np.float16, shape=(T, N))
    tokens = np.zeros(T, dtype=np.int32)
    pos = np.zeros(T, dtype=np.int16)
    win_of = np.zeros(T, dtype=np.int32)

    order = sorted(range(len(wins)), key=lambda i: -len(wins[i]))
    offsets = np.zeros(len(wins) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(w) for w in wins])
    for start in tqdm(range(0, len(order), h.batch_size), desc="harvest"):
        b = order[start : start + h.batch_size]
        L = max(len(wins[i]) for i in b)
        ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(b), L), dtype=torch.long)
        for j, i in enumerate(b):
            ids[j, : len(wins[i])] = torch.tensor(wins[i])
            mask[j, : len(wins[i])] = 1
        with torch.no_grad(), capture(model, cfg.layer, stop=True) as got:
            try:
                model.model(input_ids=ids.to(cfg.device), attention_mask=mask.to(cfg.device), use_cache=False)
            except StopForward:
                pass
        a = got["mlp_act"][:, :, idx].float()
        assert torch.isfinite(a).all()
        a = a.half().cpu().numpy()
        for j, i in enumerate(b):
            n, o = len(wins[i]), offsets[i]
            acts[o : o + n] = a[j, :n]
            tokens[o : o + n] = wins[i]
            pos[o : o + n] = np.arange(n)
            win_of[o : o + n] = i
    acts.flush()
    np.save(rd / "tokens.npy", tokens)
    np.save(rd / "pos.npy", pos)
    np.save(rd / "window.npy", win_of)
    (rd / "windows.json").write_text(json.dumps(meta))
    print(f"[harvest] done, {time.time() - t0:.0f}s")

    # Percentiles and examples, per neuron, on real tokens past skip_first.
    valid = torch.from_numpy(pos >= h.skip_first)
    nvalid = int(valid.sum())
    qs = h.percentiles
    thresholds = np.zeros((N, len(qs)), dtype=np.float32)
    rng = np.random.default_rng(h.seed)
    hit_samples, examples = {}, {}

    names = [c.name for c in h.corpora]
    corpus_of_win = np.array([names.index(m_[0]) for m_ in meta])

    def context(t):
        w0 = offsets[win_of[t]]
        lo, hi = max(w0, t - 60), min(offsets[win_of[t] + 1], t + 9)
        return {
            "corpus": meta[win_of[t]][0],
            "before": tok.decode(tokens[lo:t].tolist()),
            "token": tok.decode([int(tokens[t])]),
            "after": tok.decode(tokens[t + 1 : hi].tolist()),
        }

    for c0 in tqdm(range(0, N, h.percentile_chunk), desc="percentiles"):
        cols = torch.from_numpy(np.asarray(acts[:, c0 : c0 + h.percentile_chunk], dtype=np.float32))
        cols = cols[valid].to(cfg.device)
        for k, q in enumerate(qs):
            kth = max(1, int(round(q * nvalid)))
            thresholds[c0 : c0 + cols.shape[1], k] = cols.kthvalue(kth, dim=0).values.cpu().numpy()
        valid_idx = torch.nonzero(valid).flatten().numpy()
        for j in range(cols.shape[1]):
            n = c0 + j
            v = cols[:, j].cpu().numpy()
            th = thresholds[n]
            above = np.nonzero(v >= th[0])[0]
            take = above if len(above) <= h.hits_per_neuron else rng.choice(above, h.hits_per_neuron, replace=False)
            top = above[np.argsort(-v[above])[: h.top_k]]
            hit_samples[n] = np.union1d(take, top)
            bands = {}
            edges = list(th) + [np.inf]
            for b_ in range(len(qs)):
                inb = np.nonzero((v >= edges[b_]) & (v < edges[b_ + 1]))[0]
                pick = inb if len(inb) <= h.examples_per_band else rng.choice(inb, h.examples_per_band, replace=False)
                bands[f"p{qs[b_] * 100:g}"] = [{"act": float(v[i]), **context(valid_idx[i])} for i in pick]
            bands["top"] = [{"act": float(v[i]), **context(valid_idx[i])} for i in top[: h.examples_per_band]]
            cnt = np.bincount(corpus_of_win[win_of[valid_idx[above]]], minlength=len(h.corpora))
            by_corpus = {c.name: int(k) for c, k in zip(h.corpora, cnt)}
            examples[neurons[n]] = {"bands": bands, "p97_hits_by_corpus": by_corpus}
            hit_samples[n] = valid_idx[hit_samples[n]]

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "neurons": neurons,
        "controls": controls,
        "groups": groups,
        "percentiles": qs,
        "thresholds": {neurons[i]: thresholds[i].tolist() for i in range(N)},
        "tokens": T,
        "valid_tokens": nvalid,
        "corpora": [c.model_dump() for c in h.corpora],
        "examples": examples,
    }
    (cfg.result_dir / "harvest.json").write_text(json.dumps(out))
    save_file({str(neurons[n]): torch.from_numpy(v.astype(np.int64)) for n, v in hit_samples.items()},
              str(cfg.result_dir / "hits.safetensors"))  # fmt: skip
    print(f"[saved] harvest.json ({N} neurons), hits.safetensors; acts memmap {rd}/acts.npy {T}x{N}")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
