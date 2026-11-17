"""Stage 4b: modes of each chosen neuron's p97+ hits, by HDBSCAN on the neuron's input. GPU for the inputs.

    python scripts/modes.py --config configs/stage3_l25.yaml

1. Neurons: the top `per_group` per (emotion, neuron pole) by their summed term over the kept dims of the write,
   plus the harvest's random controls.
2. For each neuron's sampled p97+ hits (harvest hits.safetensors), the MLP input at the hit token
   (post_attention_layernorm output), re-run from the stored windows.
3. Per neuron: L2-normalise, PCA, HDBSCAN. Noise hits join the nearest cluster centroid by cosine when above that
   cluster's own `assign_quantile` of member cosines; the rest stay "other".
4. Exemplar snippets per cluster (closest to the centroid) for Haiku to name: results/<run_id>/modes/batch_XX.txt

results/<run_id>/modes.json: per neuron, clusters (size, assigned, mean act, exemplar ids) and the noise share.
"""

import json
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from tqdm import tqdm

from emotion_neurons.config import load_run_config
from emotion_neurons.model import StopForward, load_model
from emotion_neurons.selection import rank_neurons


def snippet(tokens, starts, win_of, t, tok):
    w = win_of[t]
    lo, hi = max(starts[w], t - 30), min(starts[w + 1], t + 9)  # short: Haiku reads these
    return tok.decode(tokens[lo:t].tolist()), tok.decode([int(tokens[t])]), tok.decode(tokens[t + 1 : hi].tolist())


def main():
    cfg = load_run_config(sys.argv[1:])
    m = cfg.modes
    t0 = time.time()
    rd, out_dir = cfg.run_dir, cfg.result_dir
    harvest = json.loads((out_dir / "harvest.json").read_text())
    hits = {int(k): v.numpy() for k, v in load_file(str(out_dir / "hits.safetensors")).items()}
    ns = json.loads((out_dir / "neuron_set.json").read_text())
    vec = load_file(str(cfg.results_dir / cfg.neuron_set.vectors_run / "vectors.safetensors"))
    tokens, pos, win_of = np.load(rd / "tokens.npy"), np.load(rd / "pos.npy"), np.load(rd / "window.npy")
    starts = np.zeros(win_of.max() + 2, dtype=np.int64)
    starts[1:] = np.cumsum(np.bincount(win_of, minlength=win_of.max() + 1))
    acts = np.load(rd / "acts.npy", mmap_mode="r")
    col = {n: i for i, n in enumerate(harvest["neurons"])}

    model, tok = load_model(cfg.model, cfg.device)
    mlp = model.model.layers[cfg.layer].mlp
    W = mlp.down_proj.weight.float().cpu()
    groups = rank_neurons(ns, W, vec, cfg.modes.per_group)
    groups["control"] = harvest["controls"][: m.n_controls]
    groups = {k: [n for n in v if n in col] for k, v in groups.items()}  # only harvested neurons
    chosen = sorted({n for g in groups.values() for n in g})
    print(f"[neurons] {len(chosen)} ({', '.join(f'{k}:{len(v)}' for k, v in groups.items() if k != 'control')}; controls {len(groups['control'])})")

    # MLP input at every needed hit token, from one pass over the windows that hold them.
    need = np.unique(np.concatenate([hits[n] for n in chosen]))
    by_win = defaultdict(list)
    for t in need.tolist():
        by_win[int(win_of[t])].append(t)
    wins = sorted(by_win)
    X = {}
    got = {}
    h = mlp.register_forward_pre_hook(lambda _m, a: got.__setitem__("x", a[0].detach()))
    stop = model.model.layers[cfg.layer].register_forward_hook(lambda *_: (_ for _ in ()).throw(StopForward()))
    try:
        for s in tqdm(range(0, len(wins), cfg.harvest.batch_size), desc="inputs"):
            b = wins[s : s + cfg.harvest.batch_size]
            L = max(starts[w + 1] - starts[w] for w in b)
            ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(b), L), dtype=torch.long)
            for j, w in enumerate(b):
                seq = tokens[starts[w] : starts[w + 1]]
                ids[j, : len(seq)] = torch.from_numpy(seq.astype(np.int64))
                mask[j, : len(seq)] = 1
            with torch.no_grad():
                try:
                    model.model(input_ids=ids.to(cfg.device), attention_mask=mask.to(cfg.device), use_cache=False)
                except StopForward:
                    pass
            x = got["x"].float()
            for j, w in enumerate(b):
                for t in by_win[w]:
                    X[t] = x[j, t - starts[w]].cpu().half()
    finally:
        h.remove()
        stop.remove()
    # Check: the recomputed neuron activation at the hit equals the harvested one.
    t = need[0]
    n = next(n for n in chosen if t in set(hits[n].tolist()))
    xx = X[t].float().to(cfg.device)
    a = (torch.nn.functional.silu(mlp.gate_proj.weight[n].float() @ xx) * (mlp.up_proj.weight[n].float() @ xx)).item()
    print(f"[inputs] {len(X)} hit tokens, {time.time() - t0:.0f}s | check a_{n}: recomputed {a:.3f} vs harvested {float(acts[t, col[n]]):.3f}")
    assert abs(a - float(acts[t, col[n]])) < 0.05 * max(1.0, abs(a)), "recomputed activation does not match the harvest"

    # Cluster each neuron's hits.
    result, exemplars = {}, {}
    for n in tqdm(chosen, desc="cluster"):
        ts = hits[n]
        Z = torch.stack([X[t] for t in ts.tolist()]).float()
        Z = torch.nn.functional.normalize(Z, dim=1).numpy()
        if len(ts) < 2 * m.min_cluster_size:
            lab = np.full(len(ts), -1)
        else:
            P = PCA(n_components=min(m.pca_dims, len(ts) - 1, Z.shape[1]), random_state=0).fit_transform(Z)
            lab = HDBSCAN(min_cluster_size=m.min_cluster_size, min_samples=m.min_samples, copy=True).fit_predict(P)
        a_n = np.asarray(acts[ts, col[n]], dtype=np.float32)
        clusters, final = [], lab.copy()
        cents = {}
        for c in sorted(set(lab) - {-1}):
            cen = Z[lab == c].mean(0)
            cen /= np.linalg.norm(cen)
            cents[c] = cen
        if cents:
            C = np.stack(list(cents.values()))
            cos = Z @ C.T
            keys = list(cents)
            cut = {c: np.quantile(cos[lab == c, i], m.assign_quantile) for i, c in enumerate(keys)}
            for i in np.nonzero(lab == -1)[0]:
                best = int(cos[i].argmax())
                if cos[i, best] >= cut[keys[best]]:
                    final[i] = keys[best]
        for i, c in enumerate(sorted(cents)):
            members = np.nonzero(lab == c)[0]
            close = members[np.argsort(-(Z[members] @ cents[c]))[: m.exemplars]]
            clusters.append({"id": int(c), "size": int(len(members)), "with_assigned": int((final == c).sum()),
                             "mean_act": float(a_n[final == c].mean()), "exemplars": [int(ts[i]) for i in close]})  # fmt: skip
        noise = np.nonzero(final == -1)[0]
        rng = np.random.default_rng(n)
        noise_ex = rng.choice(noise, min(m.noise_exemplars, len(noise)), replace=False) if len(noise) else []
        result[n] = {"n_hits": int(len(ts)), "clusters": clusters, "other": int(len(noise)),
                     "other_exemplars": [int(ts[i]) for i in noise_ex], "groups": [g for g, v in groups.items() if n in v]}  # fmt: skip
        for c in clusters:
            exemplars[(n, c["id"])] = c["exemplars"]
        exemplars[(n, "other")] = result[n]["other_exemplars"]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "modes.json").write_text(json.dumps({"groups": groups, "neurons": {str(k): v for k, v in result.items()}}))
    md = out_dir / "modes"
    md.mkdir(exist_ok=True)
    n_tok = 0
    for b, s in enumerate(range(0, len(chosen), m.neurons_per_batch)):
        lines = []
        for n in chosen[s : s + m.neurons_per_batch]:
            lines.append(f"##### neuron {n}")
            for key in [c["id"] for c in result[n]["clusters"]] + ["other"]:
                ex = exemplars[(n, key)]
                if not ex:
                    continue
                lines.append(f"--- cluster {key}" if key != "other" else "--- unclustered (sample)")
                for t in ex:
                    before, token, after = snippet(tokens, starts, win_of, t, tok)
                    lines.append(f"  [{float(acts[t, col[n]]):.2f}] {before}[[{token}]]{after}".replace("\n", " "))
        text = "\n".join(lines) + "\n"
        n_tok += len(text) // 4
        (md / f"batch_{b:02d}.txt").write_text(text)
    sizes = [len(r["clusters"]) for r in result.values()]
    print(f"[clusters] median {np.median(sizes):.0f} per neuron, {sum(s == 0 for s in sizes)} neurons with none; "
          f"median other share {np.median([r['other'] / r['n_hits'] for r in result.values()]):.2f}")  # fmt: skip
    print(f"[saved] modes.json, {len(list(md.glob('batch_*.txt')))} batch files, ~{n_tok} tokens for Haiku")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
