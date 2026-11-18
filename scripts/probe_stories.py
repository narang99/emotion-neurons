"""Do the harvested neurons hit their p97 band, and which of their modes, on a new set of stories? GPU.

    python scripts/probe_stories.py --config configs/adult_probe_l25.yaml

Needs results/<source_run>/{harvest,modes}.json, modes/names_*.jsonl and runs/<source_run>/{tokens,pos,window,acts}.npy
(bash scripts/fetch_runs.sh <source_run>).

1. Mode centres: modes.json keeps, per cluster, the 5 hit tokens closest to its centroid. Their MLP inputs
   (post_attention_layernorm output), L2-normalised and averaged, stand in for the centre. Leave-one-out: each
   exemplar is assigned to the nearest centre built without it (accuracy printed), and its cosine to its own
   cluster's centre gives that cluster's threshold (the lowest of its 5).
2. Stories: every token past `skip_first` where a neuron is at or above its harvest p97 is a hit. A hit goes to the
   nearest mode centre; it is "in" that mode when its cosine reaches the mode's threshold, else "other".

results/<run_id>/: hits.json (every hit: neuron, story, token, context, act/p97, mode, cosine), summary.md.
"""

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from emotion_neurons.config import load_run_config
from emotion_neurons.model import StopForward, load_model


def mlp_inputs_and_acts(model, layer, ids, mask, idx):
    """MLP input [B, L, d] and the chosen neurons' activations [B, L, N] of one batch, stopping after `layer`."""
    blk = model.model.layers[layer]
    got = {}
    hs = [blk.mlp.register_forward_pre_hook(lambda _m, a: got.__setitem__("x", a[0].detach())),
          blk.mlp.down_proj.register_forward_pre_hook(lambda _m, a: got.__setitem__("a", a[0].detach())),
          blk.register_forward_hook(lambda *_: (_ for _ in ()).throw(StopForward()))]  # fmt: skip
    try:
        with torch.no_grad():
            try:
                model.model(input_ids=ids, attention_mask=mask, use_cache=False)
            except StopForward:
                pass
    finally:
        for h in hs:
            h.remove()
    return got["x"].float(), got["a"][:, :, idx].float()


def main():
    cfg = load_run_config(sys.argv[1:])
    p = cfg.probe
    t0 = time.time()
    src_res, src_run = cfg.results_dir / p.source_run, cfg.runs_dir / p.source_run
    harvest = json.loads((src_res / "harvest.json").read_text())
    modes = json.loads((src_res / "modes.json").read_text())["neurons"]
    names = {}
    for f in sorted((src_res / "modes").glob("names_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "cluster" in r:
                    names[(int(r["neuron"]), str(r["cluster"]))] = r
    neurons = [n for n in harvest["neurons"] if str(n) in modes]  # harvest controls have no modes
    col = {n: i for i, n in enumerate(harvest["neurons"])}
    p97 = torch.tensor([harvest["thresholds"][str(n)][0] for n in neurons])
    idx = torch.tensor(neurons)  # model neuron ids (for the forward pass)
    cols = np.array([col[n] for n in neurons])  # their columns in the harvest's acts.npy
    model, tok = load_model(cfg.model, cfg.device)
    dev = cfg.device
    print(f"[load] {len(neurons)} neurons with modes from {p.source_run}, {time.time() - t0:.0f}s")

    # 1. Mode centres from the exemplars' MLP inputs.
    tokens, win_of = np.load(src_run / "tokens.npy"), np.load(src_run / "window.npy")
    acts = np.load(src_run / "acts.npy", mmap_mode="r")
    starts = np.zeros(win_of.max() + 2, dtype=np.int64)
    starts[1:] = np.cumsum(np.bincount(win_of, minlength=win_of.max() + 1))
    ex = {(n, c["id"]): c["exemplars"] for n in neurons for c in modes[str(n)]["clusters"] if c["exemplars"]}
    need = sorted({t for v in ex.values() for t in v})
    by_win = defaultdict(list)
    for t in need:
        by_win[int(win_of[t])].append(t)
    wins, X, check = sorted(by_win), {}, []
    for s in tqdm(range(0, len(wins), 32), desc="exemplars"):
        b = wins[s : s + 32]
        L = max(starts[w + 1] - starts[w] for w in b)
        ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(b), L), dtype=torch.long)
        for j, w in enumerate(b):
            seq = tokens[starts[w] : starts[w + 1]]
            ids[j, : len(seq)] = torch.from_numpy(seq.astype(np.int64))
            mask[j, : len(seq)] = 1
        x, a = mlp_inputs_and_acts(model, cfg.layer, ids.to(dev), mask.to(dev), idx.to(dev))
        for j, w in enumerate(b):
            for t in by_win[w]:
                X[t] = x[j, t - starts[w]].cpu()
                check.append((float(a[j, t - starts[w]].abs().max()), float(np.abs(acts[t][cols]).max())))
    gap = max(abs(u - v) / max(1.0, abs(v)) for u, v in check)
    print(f"[exemplars] {len(X)} tokens, {len(ex)} modes | activation check vs harvest: max rel gap {gap:.3f}")
    assert gap < 0.05, "recomputed activations do not match the harvest"

    unit = lambda v: v / v.norm(dim=-1, keepdim=True)  # noqa: E731
    cent, thr, loo_ok, loo_n = {}, {}, 0, 0
    for n in neurons:
        keys = [k for k in ex if k[0] == n]
        E = {k: unit(torch.stack([X[t] for t in ex[k]])) for k in keys}
        for k in keys:
            cent[k] = unit(E[k].mean(0))
        for k in keys:
            own = []
            for i in range(len(ex[k])):
                rest = unit(torch.cat([E[k][:i], E[k][i + 1 :]]).mean(0)) if len(ex[k]) > 1 else cent[k]
                sims = {kk: float(E[k][i] @ (rest if kk == k else cent[kk])) for kk in keys}
                loo_ok += max(sims, key=sims.get) == k
                loo_n += 1
                own.append(sims[k])
            thr[k] = min(own)
    print(f"[modes] leave-one-out: exemplars assigned to their own mode {loo_ok}/{loo_n} ({loo_ok / max(1, loo_n):.0%})")

    # 2. Stories.
    stories = []
    for f in sorted(Path(p.stories_dir).glob("*.jsonl")):
        stories += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    stories.sort(key=lambda s: s["id"])
    enc = [tok(s["text"])["input_ids"] for s in stories]
    print(f"[stories] {len(stories)} from {p.stories_dir}, {sum(map(len, enc))} tokens")
    keys_of = {n: [k for k in cent if k[0] == n] for n in neurons}
    hits, n_valid = [], 0
    for s0 in tqdm(range(0, len(stories), 16), desc="stories"):
        b = list(range(s0, min(s0 + 16, len(stories))))
        L = max(len(enc[i]) for i in b)
        ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(b), L), dtype=torch.long)
        for j, i in enumerate(b):
            ids[j, : len(enc[i])] = torch.tensor(enc[i])
            mask[j, : len(enc[i])] = 1
        x, a = mlp_inputs_and_acts(model, cfg.layer, ids.to(dev), mask.to(dev), idx.to(dev))
        x, a = unit(x).cpu(), a.cpu()
        for j, i in enumerate(b):
            T = len(enc[i])
            n_valid += T - p.skip_first
            for t, k in torch.nonzero(a[j, p.skip_first : T] >= p97).tolist():
                t += p.skip_first
                n = neurons[k]
                sims = {kk: float(x[j, t] @ cent[kk]) for kk in keys_of[n]}
                best = max(sims, key=sims.get) if sims else None
                inside = best is not None and sims[best] >= thr[best]
                nm = names.get((n, str(best[1])), {}) if inside else names.get((n, "other"), {})
                hits.append({
                    "neuron": n, "story": stories[i]["id"], "emotion": stories[i]["emotion"],
                    "explicit": stories[i].get("explicit"), "form": stories[i].get("form"), "pos": t,
                    "act_over_p97": round(float(a[j, t, k] / p97[k]), 2),
                    "before": tok.decode(enc[i][max(0, t - p.context_before) : t]), "token": tok.decode([enc[i][t]]),
                    "after": tok.decode(enc[i][t + 1 : t + 1 + p.context_after]),
                    "mode": best[1] if inside else "other", "cos": round(sims[best], 3) if best else None,
                    "mode_threshold": round(thr[best], 3) if best else None,
                    "mode_name": nm.get("mode"), "mode_emotion": nm.get("emotion"),
                })  # fmt: skip
    rate = len(hits) / max(1, n_valid * len(neurons))
    print(f"[hits] {len(hits)} neuron-token hits >= p97 over {n_valid} tokens x {len(neurons)} neurons = {rate:.3%} "
          "(harvest: 3% by construction)")  # fmt: skip

    # Summary.
    out = cfg.result_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "hits.json").write_text(json.dumps({"neurons": neurons, "valid_tokens": n_valid, "hits": hits}))
    by_n = defaultdict(list)
    for h in hits:
        by_n[h["neuron"]].append(h)
    emo_share = {}
    for n in neurons:
        m = modes[str(n)]
        lab = [names.get((n, str(c["id"])), {}).get("emotion") for c in m["clusters"]]
        emo_share[n] = sum(c["with_assigned"] for c, l_ in zip(m["clusters"], lab) if l_ and l_ != "none") / m["n_hits"]
    lines = [f"# {p.source_run} neurons on {p.stories_dir}", "",
             f"{len(stories)} stories, {n_valid} tokens. A hit: activation >= the neuron's harvest p97. Mode: nearest "
             "mode centre (from the 5 exemplars) if the cosine reaches the mode's threshold, else other. "
             "Harvest emotion share: hits in emotion-labelled modes in the harvest.", ""]  # fmt: skip
    for n in sorted(neurons, key=lambda n: -emo_share[n]):
        hs = by_n[n]
        per_tok = len(hs) / max(1, n_valid)
        inmode = Counter(h["mode"] for h in hs)
        emo_hits = sum(1 for h in hs if h["mode_emotion"] not in (None, "none"))
        lines.append(f"## {n}: harvest emotion share {emo_share[n]:.0%} | {len(hs)} hits ({per_tok:.1%} of tokens), "
                     f"{emo_hits} in emotion modes")  # fmt: skip
        by_e = Counter(h["emotion"] for h in hs)
        lines.append("- by story emotion: " + ", ".join(f"{e} {c}" for e, c in by_e.most_common()))
        for mid, c in inmode.most_common():
            nm = names.get((n, str(mid)), {})
            lines.append(f"  - {c} [{nm.get('emotion', '-')}] {nm.get('mode', 'other')}")
        for h in sorted(hs, key=lambda h: -h["act_over_p97"])[: p.examples_per_neuron]:
            ctx = f"{h['before']}[[{h['token']}]]{h['after']}".replace("\n", " ")
            lines.append(f"  - {h['act_over_p97']:.1f}x p97, {h['story']}, mode {h['mode']}: {ctx}")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines))
    print(f"[saved] {out}/hits.json, summary.md")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
