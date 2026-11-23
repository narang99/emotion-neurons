"""Stage 2b: emotion vectors from the pooled sums, plus the checks that need only the sums. CPU.

    python scripts/build_vectors.py --config configs/stage2_l25.yaml

- Reads runs/<run_id>/{sums.safetensors, stories.json}.
- Writes to results/<run_id>/:
  - vectors.safetensors (see `emotion_neurons.vectors.EmotionVectors`);
  - vectors_meta.json: emotions, sites, PCs removed per site, counts per split;
  - checks.json: nearest-vector labels for every story, split-half cosine, emotion x emotion cosine.
"""

import json
import sys

from emotion_neurons.capture import late_means
from emotion_neurons.config import VectorsRunConfig, load_run_config
from emotion_neurons.emotions import EMOTIONS, SITES
from emotion_neurons.texts import StoryRecord, StorySplit, load_story_records
from emotion_neurons.vectors import RECIPES, DenoiseInfo, EmotionVectors, cosine_matrix, remove
from emotion_neurons.vectors.checks import confusion, nearest_vector_scores, split_half_cosine


def site_checks(x, split: StorySplit, vectors: EmotionVectors, site: str, records: list[StoryRecord]) -> dict:
    """Denoised vectors for the labels; raw accuracy kept for comparison."""
    pcs = vectors.neutral_pcs(site).double()
    raw_correct = nearest_vector_scores(x, split).argmax(1) == split.label
    scores = nearest_vector_scores(x, split, pcs)
    top2 = scores.topk(2, dim=1)
    predicted = top2.indices[:, 0]
    correct = predicted == split.label
    emotional = split.emotional
    base = vectors.get(site, "mean", "raw").double()
    return {
        "acc_train_loo": correct[emotional & split.train].double().mean().item(),
        "acc_heldout": correct[emotional & split.heldout].double().mean().item(),
        "acc_train_loo_raw": raw_correct[emotional & split.train].double().mean().item(),
        "acc_heldout_raw": raw_correct[emotional & split.heldout].double().mean().item(),
        "confusion_all": confusion(predicted, split).tolist(),
        "split_half_cos": dict(zip(EMOTIONS, split_half_cosine(x, split))),
        "cos_mean_raw": cosine_matrix(base, base).tolist(),
        "cos_mean_denoised": cosine_matrix(remove(base, pcs), remove(base, pcs)).tolist(),
        "pred": {
            records[i].id: [EMOTIONS[top2.indices[i, 0]], EMOTIONS[top2.indices[i, 1]],
                            round(top2.values[i, 0].item() - scores[i, split.label[i]].item(), 4)]
            for i in emotional.nonzero().flatten().tolist()
        },
    }  # fmt: skip


def flagged_stories(checks: dict, records: list[StoryRecord], split: StorySplit) -> list[dict]:
    """Mislabelled at every site, sorted by the mean margin of the wrong vector over the story's own."""
    flagged = []
    for i in split.emotional.nonzero().flatten().tolist():
        r = records[i]
        preds = {s: checks["sites"][s]["pred"][r.id] for s in SITES}
        wrong = {s: p[0] for s, p in preds.items() if p[0] != r.emotion}
        if len(wrong) == len(SITES):
            margin = sum(p[2] for p in preds.values()) / len(SITES)
            flagged.append(
                {"id": r.id, "label": r.emotion, "heldout": r.heldout, "margin": round(margin, 4), "pred_by_site": wrong}
            )
    return sorted(flagged, key=lambda f: -f["margin"])


def vectors_meta(cfg: VectorsRunConfig, split: StorySplit, denoise: dict[str, DenoiseInfo], extract_commit: str) -> dict:
    def count(mask, e):
        return int((mask & (split.label == e)).sum())

    return {
        "emotions": list(EMOTIONS),
        "sites": list(SITES),
        "keys": "<site>/<mean|neutral>/<raw|denoised> [12, dim]; <site>/neutral_pcs; <site>/grand_mean; <site>/neutral_mean",
        "average": f"mean over tokens {cfg.data.split_token}+ per story, then over stories",
        "denoise": {site: info.model_dump() for site, info in denoise.items()},
        "n_train": {e: count(split.train, i) for i, e in enumerate(EMOTIONS)} | {"neutral": int((split.train & split.neutral).sum())},
        "n_heldout": {e: count(split.heldout, i) for i, e in enumerate(EMOTIONS)} | {"neutral": int((split.heldout & split.neutral).sum())},
        "model": cfg.model,
        "layer": cfg.layer,
        "extract_git_commit": extract_commit,
    }  # fmt: skip


def main():
    cfg = load_run_config(VectorsRunConfig, sys.argv[1:])
    records = load_story_records(cfg.run_dir)
    split = StorySplit.of(records)
    means = late_means(cfg.run_dir)
    vectors, denoise = RECIPES[cfg.vectors.recipe](means, split, cfg.vectors)

    checks = {"sites": {}}
    for site in SITES:
        checks["sites"][site] = c = site_checks(means[site], split, vectors, site, records)
        halves = sorted(c["split_half_cos"].values())
        print(f"[{site}] PCs {denoise[site].k} ({denoise[site].variance_explained:.2f} var) | denoised acc train-LOO "
              f"{c['acc_train_loo']:.3f} heldout {c['acc_heldout']:.3f} (raw {c['acc_train_loo_raw']:.3f} / "
              f"{c['acc_heldout_raw']:.3f}) | split-half cos min {halves[0]:.3f} median {halves[6]:.3f}")  # fmt: skip
    checks["flagged"] = flagged_stories(checks, records, split)
    checks["emotions"] = list(EMOTIONS)
    print(f"[flagged] {len(checks['flagged'])} stories mislabelled at all {len(SITES)} sites (denoised vectors)")

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    vectors.save(cfg.result_dir / "vectors.safetensors")
    extract_commit = json.loads((cfg.run_dir / "stories.json").read_text())["git_commit"]
    (cfg.result_dir / "vectors_meta.json").write_text(json.dumps(vectors_meta(cfg, split, denoise, extract_commit), indent=1))
    (cfg.result_dir / "checks.json").write_text(json.dumps(checks))
    print(f"[saved] {cfg.result_dir}/vectors.safetensors ({len(vectors.tensors)} arrays), vectors_meta.json, checks.json")


if __name__ == "__main__":
    main()
