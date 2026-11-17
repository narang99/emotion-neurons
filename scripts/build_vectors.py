"""Stage 2b: emotion vectors from the pooled sums, plus the checks that need only the sums. CPU.

    python scripts/build_vectors.py --config configs/stage2_l25.yaml

Reads runs/<run_id>/sums.safetensors. Writes to results/<run_id>/:
- vectors.safetensors: `<site>/<baseline>/<variant>` [12, dim] fp32, rows in EMOTIONS order;
  `<site>/neutral_pcs` [k, dim]; `<site>/grand_mean` and `<site>/neutral_mean` [dim]
- vectors_meta.json: emotions, sites, how many PCs each site removed, counts per split
- checks.json: nearest-vector labels for every story, split-half cosine, emotion x emotion cosine
"""

import json
import sys

import torch
from safetensors.torch import load_file, save_file

from emotion_neurons.config import EMOTIONS, SITES, load_run_config


def story_means(sums, counts):
    """Mean from split_token on, per story and site. mlp_out = resid_post - resid_mid, exact."""
    late = counts[:, 1:2].double()
    m = {s: sums[f"{s}/late"].double() / late for s in ("resid_mid", "resid_post", "mlp_act")}
    m["mlp_out"] = m["resid_post"] - m["resid_mid"]
    return m


def emotion_means(X, labels, mask):
    return torch.stack([X[mask & (labels == e)].mean(0) for e in range(len(EMOTIONS))])


def neutral_pcs(Xn, share):
    """Top principal directions of neutral stories explaining `share` of their variance."""
    C = Xn - Xn.mean(0)
    _, S, Vh = torch.linalg.svd(C, full_matrices=False)
    var = S**2 / (S**2).sum()
    k = int((var.cumsum(0) < share).sum().item()) + 1
    return Vh[:k], var[:k].sum().item()


def remove(V, pcs):
    return V - (V @ pcs.T) @ pcs


def cos(a, b):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return a @ b.T


def nearest_vector(X, labels, train, emo):
    """12-way label for each emotional story: the emotion whose vector (mean baseline, raw) is closest
    in cosine to the story's mean minus the grand mean. A train story is left out of its own emotion's mean."""
    mu = emotion_means(X, labels, train)
    n = torch.stack([(train & (labels == e)).sum() for e in range(len(EMOTIONS))]).double()
    scores = torch.zeros(len(X), len(EMOTIONS), dtype=torch.float64)
    for i in torch.nonzero(emo).flatten().tolist():
        m = mu.clone()
        if train[i]:
            e = labels[i]
            m[e] = (mu[e] * n[e] - X[i]) / (n[e] - 1)
        g = m.mean(0)
        scores[i] = cos((X[i] - g)[None], m - g)[0]
    return scores


def main():
    cfg = load_run_config(sys.argv[1:])
    meta = json.loads((cfg.run_dir / "stories.json").read_text())
    stories = meta["stories"]
    sums = {k: v for k, v in load_file(str(cfg.run_dir / "sums.safetensors")).items()}
    counts = sums.pop("counts")
    means = story_means(sums, counts)

    label_of = {e: i for i, e in enumerate(EMOTIONS)}
    labels = torch.tensor([label_of.get(s["emotion"], -1) for s in stories])
    heldout = torch.tensor([s["heldout"] for s in stories])
    neutral = labels == -1
    emo = ~neutral
    train = ~heldout
    topic = torch.tensor([s["topic_id"] for s in stories])

    vectors, n_pcs, checks = {}, {}, {"sites": {}}
    for site in SITES:
        X = means[site]
        mu = emotion_means(X, labels, train)
        grand = mu.mean(0)
        mu_neu = X[train & neutral].mean(0)
        pcs, explained = neutral_pcs(X[train & neutral], cfg.vectors.denoise_variance)
        n_pcs[site] = {"k": pcs.shape[0], "variance_explained": explained}
        base = {"mean": mu - grand, "neutral": mu - mu_neu}
        for b, V in base.items():
            vectors[f"{site}/{b}/raw"] = V
            vectors[f"{site}/{b}/denoised"] = remove(V, pcs)
        vectors[f"{site}/neutral_pcs"] = pcs
        vectors[f"{site}/grand_mean"] = grand
        vectors[f"{site}/neutral_mean"] = mu_neu

        # Nearest-vector label for every emotional story.
        scores = nearest_vector(X, labels, train, emo)
        top2 = scores.topk(2, dim=1)
        pred = top2.indices[:, 0]
        correct = pred == labels
        conf = torch.zeros(len(EMOTIONS), len(EMOTIONS), dtype=torch.int64)
        for i in torch.nonzero(emo).flatten().tolist():
            conf[labels[i], pred[i]] += 1

        # Split-half: vectors from topics 0-9 vs 10-19 (train stories only).
        halves = []
        for lo in (0, 10):
            h = train & (topic >= lo) & (topic < lo + 10)
            m = emotion_means(X, labels, h)
            halves.append(m - m.mean(0))
        split_half = torch.diagonal(cos(halves[0], halves[1])).tolist()

        checks["sites"][site] = {
            "acc_train_loo": correct[emo & train].double().mean().item(),
            "acc_heldout": correct[emo & heldout].double().mean().item(),
            "confusion_all": conf.tolist(),
            "split_half_cos": dict(zip(EMOTIONS, split_half)),
            "cos_mean_raw": cos(base["mean"], base["mean"]).tolist(),
            "cos_mean_denoised": cos(remove(base["mean"], pcs), remove(base["mean"], pcs)).tolist(),
            "pred": {
                stories[i]["id"]: [EMOTIONS[top2.indices[i, 0]], EMOTIONS[top2.indices[i, 1]],
                                   round(top2.values[i, 0].item() - scores[i, labels[i]].item(), 4)]
                for i in torch.nonzero(emo).flatten().tolist()
            },
        }  # fmt: skip
        c = checks["sites"][site]
        print(f"[{site}] PCs {pcs.shape[0]} ({explained:.2f} var) | acc train-LOO {c['acc_train_loo']:.3f} "
              f"heldout {c['acc_heldout']:.3f} | split-half cos min {min(split_half):.3f} median {sorted(split_half)[6]:.3f}")  # fmt: skip

    # Flag a story if it is mislabelled at 2 or more of the 4 sites.
    flagged = []
    for i in torch.nonzero(emo).flatten().tolist():
        sid = stories[i]["id"]
        wrong = {s: checks["sites"][s]["pred"][sid][0] for s in SITES if checks["sites"][s]["pred"][sid][0] != stories[i]["emotion"]}
        if len(wrong) >= 2:
            flagged.append({"id": sid, "label": stories[i]["emotion"], "heldout": stories[i]["heldout"], "pred_by_site": wrong})
    checks["flagged"] = flagged
    checks["emotions"] = list(EMOTIONS)
    print(f"[flagged] {len(flagged)} stories mislabelled at >=2 sites")

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.float().contiguous() for k, v in vectors.items()}, str(cfg.result_dir / "vectors.safetensors"))
    vmeta = {
        "emotions": list(EMOTIONS),
        "sites": list(SITES),
        "keys": "<site>/<mean|neutral>/<raw|denoised> [12, dim]; <site>/neutral_pcs; <site>/grand_mean; <site>/neutral_mean",
        "average": f"mean over tokens {cfg.data.split_token}+ per story, then over stories",
        "denoise": n_pcs,
        "n_train": {e: int((train & (labels == i)).sum()) for i, e in enumerate(EMOTIONS)} | {"neutral": int((train & neutral).sum())},
        "n_heldout": {e: int((heldout & (labels == i)).sum()) for i, e in enumerate(EMOTIONS)} | {"neutral": int((heldout & neutral).sum())},
        "model": cfg.model,
        "layer": cfg.layer,
        "extract_git_commit": meta["git_commit"],
    }  # fmt: skip
    (cfg.result_dir / "vectors_meta.json").write_text(json.dumps(vmeta, indent=1))
    (cfg.result_dir / "checks.json").write_text(json.dumps(checks))
    print(f"[saved] {cfg.result_dir}/vectors.safetensors ({len(vectors)} arrays), vectors_meta.json, checks.json")


if __name__ == "__main__":
    main()
