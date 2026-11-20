"""Is the emotion direction a sum of neurons sitting at their poles? Per-token sparse sets on raw activations. GPU.

    python scripts/raw_sets.py --config configs/raw_sets.yaml

- Tokens: per emotion, the held-out tokens (50+) of its own stories where its layer-25 score is highest.
  - Score: `(resid_post - centre) . unit_e`, centre = mean over all held-out emotion-story tokens.
- Per token, greedy on the raw terms `a_n(t) w_n` toward the layer-25 emotion vector (no mean removed;
  `decomposition.raw_greedy`). The sets are read where the cosine first reaches each threshold.
- Each member's activation is placed among its general-corpus quantiles (p1 ... p99). Pole: <= p3 or >= p97.
- Controls, at the same tokens:
  - `random`: the same greedy toward a random direction;
  - `largest`: the k neurons with the largest terms `|a_n| |w_n|`, k as the emotion's set;
  - `all`: every neuron.
- Writes results/<run_id>/raw_sets.json, and the per-token orders to runs/<run_id>/orders.safetensors.
"""

import json
import sys
import time
from collections import Counter

import torch
from safetensors.torch import load_file, save_file

from emotion_neurons.config import load_run_config
from emotion_neurons.decomposition.raw_greedy import first_reaching, raw_greedy
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.models import decoder_layers, load_model
from emotion_neurons.models.qwen3 import forward_until_stop, hooks, raise_stop, store_input, store_output
from emotion_neurons.texts import load_stories

torch.set_grad_enabled(False)
sys.path.insert(0, "scripts")
from tail_check import corpus_quantiles  # noqa: E402


def bin_labels(quantiles: list[float]) -> list[str]:
    edges = [f"p{round(100 * q)}" for q in quantiles]
    return [f"<{edges[0]}"] + [f"{a}-{b}" for a, b in zip(edges, edges[1:])] + [f">={edges[-1]}"]


def story_reads(model, tokenizer, stories, layer: int, split_token: int) -> list[dict]:
    """Per story, tokens split_token+: all neuron activations and resid_post, fp16 on the CPU."""
    block = decoder_layers(model)[layer]
    out = []
    for s in stories:
        ids = tokenizer(s.text, return_tensors="pt")["input_ids"].to(model.device)
        store = {}
        with hooks([
            block.mlp.down_proj.register_forward_pre_hook(store_input(store, "a")),
            block.register_forward_hook(store_output(store, "r")),
            block.register_forward_hook(raise_stop),
        ]):  # fmt: skip
            forward_until_stop(model, ids, torch.ones_like(ids))
        out.append({
            "story": s, "ids": ids[0].tolist(),
            "a": store["a"][0, split_token:].half().cpu(), "r": store["r"][0, split_token:].half().cpu(),
        })  # fmt: skip
    return out


def top_tokens(reads: list[dict], centre, unit, n: int, per_story: int) -> list[tuple[float, int, int]]:
    """(score, read index, position in the read) of the n highest-scoring tokens, at most per_story per story."""
    picks = []
    for i, rd in enumerate(reads):
        score = (rd["r"].float() - centre) @ unit
        values, pos = score.topk(min(per_story, len(score)))
        picks += [(float(v), i, int(p)) for v, p in zip(values, pos)]
    return sorted(picks, reverse=True)[:n]


def place(a, Q, members):
    """Bin index of each member's activation among its quantiles, and its pole (+1 high, -1 low, 0 middle).

    - `a` [token, k] holds the activations of `members` [token, k]; `Q` is [quantile, neuron].
    """
    q = Q[:, members]  # [quantile, token, k]
    bins = (a.unsqueeze(0) >= q).sum(0)
    lo, hi = q[1], q[-2]  # p3 and p97 (the quantile grid starts p1, p3 and ends p97, p99)
    pole = (a >= hi).long() - (a <= lo).long()
    return bins, pole


def group_stats(a_sel, Q, members, weights, n_bins: int) -> dict:
    """Bin shares and pole shares over all (token, member) pairs; `weights` gives each member's projection."""
    a = a_sel.gather(1, members)
    bins, pole = place(a, Q, members)
    hist = torch.bincount(bins.flatten(), minlength=n_bins).float()
    w_total = weights.sum()
    return {
        "bins": (hist / hist.sum()).tolist(),
        "high": float((pole == 1).float().mean()),
        "low": float((pole == -1).float().mean()),
        "projection_high": float(weights[pole.flatten() == 1].sum() / w_total),
        "projection_low": float(weights[pole.flatten() == -1].sum() / w_total),
    }


def read_sets(a_sel, Q, W, run, d_hat, k_per_token, n_bins: int) -> dict:
    """Stats of the sets at their first-reaching sizes; tokens that never reach are left out."""
    rows = (k_per_token > 0).nonzero().squeeze(1)
    if len(rows) == 0:
        return {"tokens": 0}
    groups = []
    for i in rows.tolist():
        groups.append(run.order[i, : int(k_per_token[i])])
    stats = {"bins": torch.zeros(n_bins), "high": 0.0, "low": 0.0, "projection_high": 0.0, "projection_low": 0.0}
    for i, members in zip(rows.tolist(), groups):
        m = members[None]
        weights = a_sel[i, members] * (d_hat[i] @ W[:, members])
        s = group_stats(a_sel[i : i + 1], Q, m, weights, n_bins)
        stats["bins"] += torch.tensor(s["bins"]) * len(members)
        for key in ("high", "low"):
            stats[key] += s[key] * len(members)
        for key in ("projection_high", "projection_low"):
            stats[key] += s[key]
    n_members = sum(len(m) for m in groups)
    return {
        "tokens": len(rows), "members": n_members,
        "bins": (stats["bins"] / n_members).tolist(),
        "high": stats["high"] / n_members, "low": stats["low"] / n_members,
        "projection_high": stats["projection_high"] / len(rows), "projection_low": stats["projection_low"] / len(rows),
    }  # fmt: skip


def size_stats(k: torch.Tensor) -> dict:
    reached = k[k > 0].float()
    if len(reached) == 0:
        return {"reached": 0.0}
    q = torch.quantile(reached, torch.tensor([0.25, 0.5, 0.75], device=reached.device))
    return {"reached": float((k > 0).float().mean()), "k_q25": float(q[0]), "k_median": float(q[1]), "k_q75": float(q[2])}


def largest_terms(a_sel, W, k_per_token) -> list[torch.Tensor]:
    size = a_sel.abs() * W.norm(dim=0)
    return [size[i].topk(int(k)).indices for i, k in enumerate(k_per_token.tolist()) if k > 0]


def main():
    cfg = load_run_config(sys.argv[1:])
    rs = cfg.raw_sets
    t0 = time.time()
    model, tokenizer = load_model(cfg.model, cfg.device)
    Q, n_corpus = corpus_quantiles(model, tokenizer, cfg, rs.quantiles)
    assert abs(rs.quantiles[1] - 0.03) < 1e-9 and abs(rs.quantiles[-2] - 0.97) < 1e-9, "poles read p3 and p97 from the grid"
    Q = Q.to(cfg.device)
    labels = bin_labels(rs.quantiles)
    print(
        f"[quantiles] {Q.shape[0]} quantiles of {Q.shape[1]} neurons over {n_corpus} corpus tokens, {time.time() - t0:.0f}s",
        flush=True,
    )

    W = decoder_layers(model)[cfg.layer].mlp.down_proj.weight.float()
    WtW = W.T @ W
    units = load_file(str(rs.layer_vectors_file))["unit"][cfg.layer].float().to(cfg.device)
    sets = {e: set(v["match"]["neurons"]) for e, v in json.loads(rs.sets_file.read_text()).items()}
    stories = [s for s in load_stories(cfg.data) if s.heldout and s.emotion in EMOTIONS]
    reads = story_reads(model, tokenizer, stories, cfg.layer, cfg.data.split_token)
    centre = torch.cat([rd["r"].float() for rd in reads]).mean(0)
    print(f"[stories] {len(reads)} held-out stories, {time.time() - t0:.0f}s", flush=True)

    generator = torch.Generator().manual_seed(rs.seed)
    out, saved = {"corpus_tokens": n_corpus, "bins": labels, "emotions": {}}, {}
    for e, name in enumerate(EMOTIONS):
        own = [i for i, rd in enumerate(reads) if rd["story"].emotion == name]
        picks = top_tokens([reads[i] for i in own], centre, units[e].cpu(), rs.tokens_per_emotion, rs.per_story)
        a_sel = torch.stack([reads[own[i]]["a"][p] for _, i, p in picks]).float().to(cfg.device)
        random_dir = torch.randn(W.shape[0], generator=generator).to(cfg.device)
        row = {"tokens": len(picks), "score_min": picks[-1][0], "score_max": picks[0][0], "conditions": {}}
        for cond, d in (("emotion", units[e]), ("random", random_dir)):
            d_rows = (d / d.norm()).expand(len(picks), -1)
            run = raw_greedy(W, WtW, a_sel, d_rows, rs.steps)
            saved[f"{name}/{cond}/order"] = run.order.int().cpu()
            saved[f"{name}/{cond}/cos"] = run.cos.cpu()
            c = {"full_cos": float(run.full_cos.mean()), "max_cos": float(run.cos.max(1).values.mean()), "at": {}}
            for thr in rs.thresholds:
                k = first_reaching(run.cos, thr)
                at = {"size": size_stats(k), "sets": read_sets(a_sel, Q, W, run, d_rows, k, len(labels))}
                if cond == "emotion":
                    big = largest_terms(a_sel, W, k)
                    rows = (k > 0).nonzero().squeeze(1)
                    if len(big):
                        all_bins = torch.zeros(len(labels))
                        hi_lo = torch.zeros(2)
                        for i, m in zip(rows.tolist(), big):
                            bins, pole = place(a_sel[i, m][None], Q, m[None])
                            all_bins += torch.bincount(bins.flatten(), minlength=len(labels)).float().cpu()
                            hi_lo += torch.tensor([float((pole == 1).sum()), float((pole == -1).sum())])
                        at["largest"] = {"bins": (all_bins / all_bins.sum()).tolist(), "high": float(hi_lo[0] / all_bins.sum()),
                                         "low": float(hi_lo[1] / all_bins.sum())}  # fmt: skip
                    memberships = Counter(n for i in rows.tolist() for n in run.order[i, : int(k[i])].tolist())
                    total = sum(memberships.values())
                    at["in_match_set"] = sum(v for n, v in memberships.items() if n in sets[name]) / max(total, 1)
                    at["match_set_chance"] = len(sets[name]) / W.shape[1]
                    at["distinct_neurons"] = len(memberships)
                    at["top_neurons"] = [
                        {"neuron": n, "token_share": v / max(len(rows), 1), "in_match_set": n in sets[name]}
                        for n, v in memberships.most_common(rs.top_neurons)
                    ]
                c["at"][str(thr)] = at
            row["conditions"][cond] = c
        all_idx = torch.arange(W.shape[1], device=cfg.device).expand(len(picks), -1)
        bins, pole = place(a_sel, Q, all_idx)
        hist = torch.bincount(bins.flatten(), minlength=len(labels)).float()
        row["all_neurons"] = {"bins": (hist / hist.sum()).tolist(), "high": float((pole == 1).float().mean()),
                              "low": float((pole == -1).float().mean())}  # fmt: skip
        row["examples"] = []
        for score, i, p in picks[:5]:
            rd = reads[own[i]]
            t = cfg.data.split_token + p
            ctx = tokenizer.decode(rd["ids"][max(0, t - 12) : t]) + "[[" + tokenizer.decode(rd["ids"][t : t + 1]) + "]]"
            row["examples"].append({"score": score, "story": rd["story"].id, "context": ctx.replace("\n", " ")})
        out["emotions"][name] = row
        em, rn = row["conditions"]["emotion"], row["conditions"]["random"]
        line = f"[{name}] full cos {em['full_cos']:.2f} (random {rn['full_cos']:.2f}) | max {em['max_cos']:.2f} (random {rn['max_cos']:.2f})"
        for thr in rs.thresholds:
            a, b = em["at"][str(thr)], rn["at"][str(thr)]
            if a["sets"].get("tokens"):
                line += (f" | cos {thr}: k {a['size']['k_median']:.0f} reached {a['size']['reached']:.2f}, poles {a['sets']['high'] + a['sets']['low']:.2f}"
                         f" (random {b['sets'].get('high', 0) + b['sets'].get('low', 0):.2f}, largest {a['largest']['high'] + a['largest']['low']:.2f}),"
                         f" in match set {a['in_match_set']:.2f} (chance {a['match_set_chance']:.2f})")  # fmt: skip
        line += f" | all neurons poles {row['all_neurons']['high'] + row['all_neurons']['low']:.3f} | {time.time() - t0:.0f}s"
        print(line, flush=True)

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "raw_sets.json").write_text(json.dumps(out, indent=1))
    run_dir = cfg.runs_dir / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in saved.items()}, str(run_dir / "orders.safetensors"))
    print(f"[saved] {cfg.result_dir}/raw_sets.json, {run_dir}/orders.safetensors")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
