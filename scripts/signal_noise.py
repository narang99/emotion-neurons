"""Signal and noise neurons along each emotion vector, from raw story-level pushes; and the modes of the offset neurons. GPU.

    python scripts/signal_noise.py --config configs/signal_noise.yaml

- Push of neuron n along the layer-25 emotion unit vector v: `a_n (w_n . v)`, raw (`decomposition.signal_noise`).
- Per story: the mean push over tokens 50+, split into tokens at the high pole (>= p97), low pole (<= p3) and middle.
- Per emotion, on its 100 train stories: `signal` (mean push not zero), `noise` (the rest); signal splits into
  `specific` (differs from the other stories) and `constant` (the same on any story: an offset).
- Checks on the held-out stories: does the noise sum cancel; how much of the push each group carries; how well each
  group's push separates the emotion's stories from the others.
- Control: the same with a random direction per emotion.
- Modes: for the strongest constant and specific neurons, their general-corpus statistics and top / bottom contexts.
- Writes results/<run_id>/signal_noise.json and report.md; caches the story class means in runs/<run_id>/.
"""

import json
import sys
import time
from collections import Counter

import torch
from safetensors.torch import load_file, save_file

from emotion_neurons.config import load_run_config
from emotion_neurons.decomposition.signal_noise import classify, neurons_for_mass, noise_spread
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.harvest import collect_windows, padded_batch
from emotion_neurons.models import decoder_layers, load_model, read_mlp
from emotion_neurons.texts import load_stories

torch.set_grad_enabled(False)
sys.path.insert(0, "scripts")
from tail_check import CLASSES, corpus_quantiles, story_class_means  # noqa: E402

GROUPS = ("signal", "specific", "constant", "noise")
MODE_NEURONS = 10


def thresholds(model, tokenizer, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    f = cfg.signal_noise.thresholds_file
    if f.exists():
        cached = load_file(str(f))
        return cached["p3"], cached["p97"]
    Q, _ = corpus_quantiles(model, tokenizer, cfg, [0.03, 0.97])
    return Q[0], Q[1]


def class_means_cached(model, tokenizer, cfg, stories, lo, hi) -> tuple[dict, torch.Tensor]:
    f = cfg.runs_dir / cfg.run_id / "story_classes.safetensors"
    if f.exists():
        t = load_file(str(f))
        return {c: t[c].float() for c in CLASSES}, t["pole_share"].float()
    means, shares = story_class_means(model, tokenizer, stories, cfg.layer, lo, hi, cfg.data.split_token)
    pole_share = shares[:, 0] + shares[:, 1]
    f.parent.mkdir(parents=True, exist_ok=True)
    save_file({**{c: v.contiguous() for c, v in means.items()}, "pole_share": pole_share.half().contiguous()}, str(f))
    return means, pole_share


def d_prime(x_own: torch.Tensor, x_other: torch.Tensor) -> float:
    pooled = ((x_own.var() + x_other.var()) / 2).sqrt()
    return float((x_own.mean() - x_other.mean()) / pooled)


def analyse(k, class_means, pole_share, masks, sn) -> tuple[dict, object]:
    """One direction (`k[n] = w_n . v`) for one emotion's stories."""
    own_train, own_held, other_train, other_held = masks
    class_push = {c: class_means[c] * k for c in CLASSES}
    P = sum(class_push.values())
    g = classify(P[own_train], P[other_train], sn.t_cut)
    total = float(g.mean.sum())
    contrast = g.mean - g.mean_other
    row = {
        "counts": {},
        "share_of_mean_push": {},
        "heldout_share": {},
        "d_prime": {},
        "pole_split": {},
        "contrast_pole_split": {},
    }
    held_total = P[own_held].sum(1)
    for name in GROUPS:
        m = getattr(g, name)
        row["counts"][name] = {"n": int(m.sum()), "positive": int((m & (g.mean > 0)).sum())}
        row["share_of_mean_push"][name] = float(g.mean[m].sum()) / total
        row.setdefault("share_of_contrast", {})[name] = float(contrast[m].sum() / contrast.sum())
        row["heldout_share"][name] = float(P[own_held][:, m].sum(1).mean() / held_total.mean())
        row["d_prime"][name] = d_prime(P[own_held][:, m].sum(1), P[other_held][:, m].sum(1))
        mc = {c: class_push[c][own_train][:, m].mean(0).sum() for c in CLASSES}
        row["pole_split"][name] = {c: float(v / sum(mc.values())) for c, v in mc.items()}
        cc = {c: (class_push[c][own_train][:, m].mean(0) - class_push[c][other_train][:, m].mean(0)).sum() for c in CLASSES}
        row["contrast_pole_split"][name] = {c: float(v / sum(cc.values())) for c, v in cc.items()}
        row.setdefault("pole_token_share", {})[name] = float(pole_share[own_train][:, m].mean())
    row["d_prime"]["all"] = d_prime(held_total, P[other_held].sum(1))
    row["mean_push"] = {"own_train": total, "other_train": float(g.mean_other.sum()), "own_heldout": float(held_total.mean())}
    row["noise"] = noise_spread(P[own_train], P[own_held], g.noise)
    row["neurons_for_mass"] = {
        "signal_of_push": neurons_for_mass(g.mean, g.signal, sn.mass),
        "specific_of_contrast": neurons_for_mass(contrast, g.specific, sn.mass),
        "specific_of_own_contrast": neurons_for_mass(torch.where(g.specific, contrast, 0.0), g.specific, sn.mass),
    }
    return row, g


def corpus_modes(model, tokenizer, cfg, neurons: list[int], lo, hi) -> dict[int, dict]:
    """General-corpus statistics and top / bottom contexts of a few neurons."""
    h = cfg.harvest
    windows = collect_windows(h, tokenizer)
    idx = torch.tensor(neurons, device=model.device)
    values, where = [], []
    for start in range(0, len(windows.tokens), h.batch_size):
        ids, mask = padded_batch(windows.tokens[start : start + h.batch_size], tokenizer.pad_token_id)
        _, a = read_mlp(model, cfg.layer, ids.to(model.device), mask.to(model.device), idx)
        keep = mask.bool()
        keep[:, : h.skip_first] = False
        values.append(a[keep.to(a.device)].half().cpu())
        where += [(start + int(w), int(p)) for w, p in keep.nonzero().tolist()]
    acts = torch.cat(values).float()

    def context(w: int, p: int) -> str:
        toks = windows.tokens[w]
        return (
            tokenizer.decode(toks[max(0, p - 10) : p])
            + "[["
            + tokenizer.decode(toks[p : p + 1])
            + "]]"
            + tokenizer.decode(toks[p + 1 : p + 3])
        ).replace("\n", " ")

    out = {}
    for j, n in enumerate(neurons):
        col = acts[:, j]
        top, bottom = col.topk(200).indices, (-col).topk(200).indices
        out[n] = {
            "mean": float(col.mean()), "median": float(col.median()), "p3": float(lo[n]), "p97": float(hi[n]),
            "share_high": float((col >= hi[n]).float().mean()), "share_low": float((col <= lo[n]).float().mean()),
            "top_tokens": [t for t, _ in Counter(tokenizer.decode([windows.tokens[where[i][0]][where[i][1]]]) for i in top.tolist()).most_common(8)],
            "bottom_tokens": [t for t, _ in Counter(tokenizer.decode([windows.tokens[where[i][0]][where[i][1]]]) for i in bottom.tolist()).most_common(8)],
            "top_contexts": [context(*where[i]) for i in top[:4].tolist()],
            "bottom_contexts": [context(*where[i]) for i in bottom[:4].tolist()],
        }  # fmt: skip
    return out


def report_lines(out: dict, modes: dict) -> list[str]:
    lines = ["# Signal, noise and offset neurons along each emotion vector (layer 25, raw pushes)", ""]
    for e, r in out["emotions"].items():
        em = r["emotion"]
        c = em["counts"]
        lines += [f"## {e}", "",
                  f"- counts: specific {c['specific']['n']} ({c['specific']['positive']} push +), constant {c['constant']['n']} "
                  f"({c['constant']['positive']} +), noise {c['noise']['n']}",
                  "- share of the mean push: " + ", ".join(f"{g} {v:+.2f}" for g, v in em["share_of_mean_push"].items()),
                  "- d' (own vs other held-out stories): " + ", ".join(f"{g} {v:+.2f}" for g, v in em["d_prime"].items()),
                  f"- noise sum: mean {em['noise']['mean_train']:+.3f}, sd {em['noise']['sd_train']:.3f} "
                  f"(independent: {em['noise']['sd_if_independent']:.3f})", ""]  # fmt: skip
        for kind in ("constant", "specific"):
            lines += [f"### {e}: strongest {kind} neurons", ""]
            for item in r["mode_neurons"][kind]:
                m = modes[item["neuron"]]
                lines += [f"- **n{item['neuron']}** push {item['push']:+.3f} (other stories {item['push_other']:+.3f}); corpus median "
                          f"{m['median']:+.2f}, p3 {m['p3']:+.2f}, p97 {m['p97']:+.2f}",
                          f"  - top tokens: {', '.join(repr(t) for t in m['top_tokens'])}",
                          f"  - bottom tokens: {', '.join(repr(t) for t in m['bottom_tokens'])}",
                          f"  - high: {' / '.join(m['top_contexts'][:2])}",
                          f"  - low: {' / '.join(m['bottom_contexts'][:2])}"]  # fmt: skip
            lines.append("")
    return lines


def main():
    cfg = load_run_config(sys.argv[1:])
    sn = cfg.signal_noise
    t0 = time.time()
    model, tokenizer = load_model(cfg.model, cfg.device)
    lo, hi = thresholds(model, tokenizer, cfg)
    stories = load_stories(cfg.data)
    class_means, pole_share = class_means_cached(model, tokenizer, cfg, stories, lo, hi)
    print(f"[stories] {len(stories)} stories, {time.time() - t0:.0f}s", flush=True)

    W = decoder_layers(model)[cfg.layer].mlp.down_proj.weight.float().cpu()
    units = load_file(str(sn.layer_vectors_file))["unit"][cfg.layer].float()
    sets = {e: set(v["match"]["neurons"]) for e, v in json.loads(sn.sets_file.read_text()).items()}
    emotion = [s.emotion for s in stories]
    held = torch.tensor([s.heldout for s in stories])
    generator = torch.Generator().manual_seed(sn.seed)
    out = {"t_cut": sn.t_cut, "emotions": {}}
    for e, name in enumerate(EMOTIONS):
        own = torch.tensor([x == name for x in emotion])
        masks = (own & ~held, own & held, ~own & ~held, ~own & held)
        row, g = analyse(W.T @ units[e], class_means, pole_share, masks, sn)
        random_dir = torch.randn(W.shape[0], generator=generator)
        row_random, _ = analyse(W.T @ (random_dir / random_dir.norm()), class_means, pole_share, masks, sn)
        match = torch.zeros(W.shape[1], dtype=torch.bool)
        match[list(sets[name])] = True
        row["match_set"] = {"specific_in_match": float((g.specific & match).sum() / g.specific.sum().clamp_min(1)),
                            "match_in_specific": float((g.specific & match).sum() / match.sum()),
                            "constant_in_match": float((g.constant & match).sum() / g.constant.sum().clamp_min(1))}  # fmt: skip
        mode_neurons = {}
        for kind, score in (("constant", g.mean.abs()), ("specific", (g.mean - g.mean_other).abs())):
            members = getattr(g, kind).nonzero().squeeze(1)
            top = members[score[members].argsort(descending=True)[:MODE_NEURONS]].tolist()
            mode_neurons[kind] = [{"neuron": n, "push": float(g.mean[n]), "push_other": float(g.mean_other[n])} for n in top]
        out["emotions"][name] = {"emotion": row, "random": row_random, "mode_neurons": mode_neurons}
        c, rc = row["counts"], row_random["counts"]
        print(f"[{name}] specific {c['specific']['n']} constant {c['constant']['n']} noise {c['noise']['n']} (random dir: specific {rc['specific']['n']}, constant {rc['constant']['n']})"
              f" | share of push: " + ", ".join(f"{k} {v:+.2f}" for k, v in row["share_of_mean_push"].items())
              + " | d': " + ", ".join(f"{k} {v:+.2f}" for k, v in row["d_prime"].items())
              + f" | noise mean {row['noise']['mean_train']:+.3f} sd {row['noise']['sd_train']:.3f} (indep {row['noise']['sd_if_independent']:.3f})"
              + f" | specific poles of contrast: high {row['contrast_pole_split']['specific']['high']:+.2f} low {row['contrast_pole_split']['specific']['low']:+.2f}"
              + " | share of contrast: " + ", ".join(f"{k} {v:+.2f}" for k, v in row["share_of_contrast"].items())
              + f" | k90 of specific's contrast {row['neurons_for_mass']['specific_of_own_contrast']}"
              + f" | random dir d' all {row_random['d_prime']['all']:+.2f}", flush=True)  # fmt: skip

    neurons = sorted({i["neuron"] for r in out["emotions"].values() for kind in r["mode_neurons"].values() for i in kind})
    modes = corpus_modes(model, tokenizer, cfg, neurons, lo, hi)
    print(f"[modes] {len(neurons)} neurons, {time.time() - t0:.0f}s", flush=True)
    out["modes"] = {str(n): v for n, v in modes.items()}
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "signal_noise.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (cfg.result_dir / "report.md").write_text("\n".join(report_lines(out, modes)))
    print(f"[saved] {cfg.result_dir}/signal_noise.json, report.md")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
