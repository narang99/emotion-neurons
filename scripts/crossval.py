"""Cross-validation of the layer-25 results: 6 folds instead of one held-out split, and kappa chosen on held-out data.

    python scripts/crossval.py --config configs/crossval.yaml

- Folds: each emotion has stories with story_idx 0-5 (20 per index). Fold k holds out story_idx k, and neutral
  stories 2k and 2k+1. Stage 2's split is fold 5.
- Per fold:
  - 12-way held-out accuracy of the denoised `resid_post` vectors (the Stage 2 check, `nearest_vector_scores`);
  - per emotion and per kappa: the greedy order of the MLP write on the train folds (kappa 0: the plain target,
    else the shrunk target of `selection.shrunk`), and its held-out transfer at `sizes`; and the full set's transfer.
- kappa is chosen by the mean transfer at `choose_at` over folds and emotions.
- No forward pass: the pooled sums of Stage 2 (`runs/stage2_l25`) and layer 25's `W_down`.
- Writes results/<run_id>/crossval.json.
"""

import json
import statistics
import sys
import time

import torch

from emotion_neurons.capture import late_means
from emotion_neurons.config import load_run_config
from emotion_neurons.decomposition import greedy, prefix_curves
from emotion_neurons.decomposition import selection as sel
from emotion_neurons.decomposition.shifts import shifts_from_means
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.models import load_w_down
from emotion_neurons.texts import StorySplit, load_story_records
from emotion_neurons.vectors import RECIPES
from emotion_neurons.vectors.checks import nearest_vector_scores

torch.set_grad_enabled(False)


def fold_split(base: StorySplit, story_idx: torch.Tensor, k: int) -> StorySplit:
    held = torch.where(base.emotional, story_idx == k, (story_idx == 2 * k) | (story_idx == 2 * k + 1))
    return StorySplit(label=base.label, heldout=held, topic=base.topic)


def fold_accuracy(resid: torch.Tensor, split: StorySplit, cfg) -> float:
    vectors, _ = RECIPES[cfg.vectors.recipe]({"resid_post": resid}, split, cfg.vectors)
    scores = nearest_vector_scores(resid, split, vectors.neutral_pcs("resid_post").double())
    held = split.heldout & split.emotional
    return float((scores.argmax(1) == split.label)[held].double().mean())


def main():
    cfg = load_run_config(sys.argv[1:])
    cv = cfg.crossval
    t0 = time.time()
    run_dir = cfg.runs_dir / cv.vectors_run
    records = load_story_records(run_dir)
    base = StorySplit.of(records)
    story_idx = torch.tensor([r.story_idx for r in records])
    means = late_means(run_dir)
    W = load_w_down(cfg.model, cfg.layer).to(cfg.device)
    WtW = W.T @ W
    print(f"[load] {len(records)} stories, W {tuple(W.shape)} on {cfg.device}, {time.time() - t0:.0f}s", flush=True)

    folds = []
    for k in range(6):
        split = fold_split(base, story_idx, k)
        accuracy = fold_accuracy(means["resid_post"], split, cfg)
        shifts = shifts_from_means(means["mlp_act"].float(), split, cv.seed)
        emotions = {}
        for e, name in enumerate(EMOTIONS):
            a, a_held, se = (x[e].to(cfg.device) for x in (shifts.train, shifts.heldout, shifts.std_error))
            d = W @ a
            G = sel.gram(WtW, a)
            row = {"full_transfer": float(torch.nn.functional.cosine_similarity(W @ a_held, d, dim=0))}
            for kappa in cv.kappas:
                c = sel.plain(W, a) if kappa == 0 else sel.shrunk(W, a, se, kappa)
                order = greedy(G, c, max(cv.sizes))[0]
                _, transfer = prefix_curves(W, a, d, a_held, order)
                row[f"kappa{kappa:g}"] = {str(s): float(transfer[s - 1]) for s in cv.sizes}
            emotions[name] = row
            del G
        folds.append({"fold": k, "accuracy": accuracy, "emotions": emotions})
        at = str(cv.choose_at)
        print(f"[fold {k}] accuracy {accuracy:.3f} | transfer at {at}: "
              + ", ".join(f"k{kp:g} {statistics.mean(r[f'kappa{kp:g}'][at] for r in emotions.values()):.3f}" for kp in cv.kappas)
              + f" | full {statistics.mean(r['full_transfer'] for r in emotions.values()):.3f}, {time.time() - t0:.0f}s", flush=True)  # fmt: skip

    def over_folds(get) -> dict:
        values = [statistics.mean(get(f["emotions"][e]) for e in EMOTIONS) for f in folds]
        return {"mean": statistics.mean(values), "sd": statistics.stdev(values), "per_fold": values}

    summary = {
        "accuracy": {"mean": statistics.mean(f["accuracy"] for f in folds), "sd": statistics.stdev(f["accuracy"] for f in folds),
                     "per_fold": [f["accuracy"] for f in folds]},
        "full_transfer": over_folds(lambda r: r["full_transfer"]),
        "transfer": {f"kappa{kp:g}": {str(s): over_folds(lambda r, kp=kp, s=s: r[f"kappa{kp:g}"][str(s)]) for s in cv.sizes} for kp in cv.kappas},
    }  # fmt: skip
    best = max(cv.kappas, key=lambda kp: summary["transfer"][f"kappa{kp:g}"][str(cv.choose_at)]["mean"])
    summary["chosen_kappa"] = best
    print(f"[summary] accuracy {summary['accuracy']['mean']:.3f} +- {summary['accuracy']['sd']:.3f} | chosen kappa {best:g} "
          + " | ".join(f"k{kp:g}: " + ", ".join(f"{s} {summary['transfer'][f'kappa{kp:g}'][str(s)]['mean']:.3f}" for s in cv.sizes) for kp in cv.kappas), flush=True)  # fmt: skip
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "crossval.json").write_text(json.dumps({"summary": summary, "folds": folds}, indent=1))
    print(f"[saved] {cfg.result_dir}/crossval.json")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
