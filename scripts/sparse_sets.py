"""Sparse sets by a relaxed mask, stopped by held-out early stopping over 6 folds. CPU.

    python scripts/sparse_sets.py --config configs/sparse_relaxed_l25.yaml --device cpu

- Target per emotion: `d = W_down a`, `a = abar_own - abar_other` (`emotion_neurons.sparse.gaps`).
- Two searches, each run on the train part of every fold:
  - relaxed mask (`sparse.relaxed`): a lam path; each point's set is its support;
  - greedy on the leftover (`sparse.greedy`): the earlier method, as the baseline.
- Stop (`sparse.crossval`): the lam, or the greedy size, with the lowest held-out error averaged over the folds.
- Final sets: both searches on all stories, read at the chosen lam and size.
- Reads runs/<sparse.sums_run>/{sums.safetensors, stories.json} and the layer's W_down.
- Writes results/<run_id>/:
  - summary.json: per emotion, the fold curves, the chosen stops, the final sets' scores;
  - neurons/<emotion>.csv: one row per neuron in either final set, sorted by its share of d;
  - tensors.safetensors: `d`, `shift`, `mask`, `abar_own`, `abar_other` [emotion, dim], rows in `sparse.emotions` order.
"""

import csv
import json
import sys
import time
from pathlib import Path

import torch
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict
from safetensors.torch import save_file
from torch import Tensor

from emotion_neurons.capture import late_means
from emotion_neurons.config import SparseConfig, load_run_config
from emotion_neurons.models import load_w_down
from emotion_neurons.sparse import (
    GreedyRun,
    MaskSolution,
    activation_gap,
    emotion_groups,
    first_below,
    fold_split,
    lambda_max,
    lambda_path,
    lipschitz,
    no_heldout,
    prefix_errors,
    random_members,
    residual_greedy,
    scaled_columns,
    set_error,
    set_leftover,
    support,
)
from emotion_neurons.texts import StorySplit, load_story_records

torch.set_grad_enabled(False)


class Searches(BaseModel):
    """Both searches for one emotion on one set of train stories."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    shift: Float[Tensor, " d_mlp"]
    greedy: GreedyRun
    path: list[MaskSolution]
    lam_max: float


class FoldCurves(BaseModel):
    """Held-out error per fold: greedy by set size, relaxed by lam point; and the relaxed set sizes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    greedy: Float[Tensor, "fold step"]
    relaxed: Float[Tensor, "fold point"]
    relaxed_size: Int[Tensor, "fold point"]
    full_mlp: Float[Tensor, " fold"]


def lambda_ratios(sc: SparseConfig) -> list[float]:
    return torch.logspace(0, torch.log10(torch.tensor(sc.lambda_min_ratio)), sc.lambda_points).tolist()


def run_searches(W, WtW, A, split: StorySplit, emotion: str, sc: SparseConfig) -> Searches:
    groups = emotion_groups(split, emotion)
    shift = activation_gap(A, groups.own, groups.other)
    B, d_unit = scaled_columns(W, shift)
    path = lambda_path(B, d_unit, lambda_ratios(sc), lipschitz(B), sc.max_iterations, sc.tolerance)
    greedy = residual_greedy(W, WtW, shift, sc.greedy_steps)
    return Searches(shift=shift, greedy=greedy, path=path, lam_max=lambda_max(B, d_unit))


def heldout_target(W, A, split: StorySplit, emotion: str) -> Float[Tensor, " d_model"]:
    groups = emotion_groups(split, emotion)
    return W @ activation_gap(A, groups.own_heldout, groups.other_heldout)


def fold_curves(W, WtW, A, base: StorySplit, story_idx, emotion: str, sc: SparseConfig, t0: float) -> FoldCurves:
    greedy, relaxed, sizes, full = [], [], [], []
    everyone = torch.ones(W.shape[1], dtype=torch.bool)
    for k in range(sc.folds):
        split = fold_split(base, story_idx, k)
        s = run_searches(W, WtW, A, split, emotion, sc)
        target = heldout_target(W, A, split, emotion)
        greedy.append(prefix_errors(W, s.shift, s.greedy.order, target))
        relaxed.append(torch.tensor([set_error(W, s.shift, support(p.mask) > 0, target) for p in s.path]))
        sizes.append(torch.tensor([int(support(p.mask).sum()) for p in s.path]))
        full.append(set_error(W, s.shift, everyone, target))
        print(f"  [{emotion} fold {k}] greedy min {float(greedy[-1].min()):.3f} at {int(greedy[-1].argmin()) + 1},"
              f" relaxed min {float(relaxed[-1].min()):.3f} at {int(sizes[-1][relaxed[-1].argmin()])}, all {full[-1]:.3f}"
              f" | {time.time() - t0:.0f}s", flush=True)  # fmt: skip
    steps = min(len(g) for g in greedy)  # greedy can stop early when no neuron helps
    return FoldCurves(
        greedy=torch.stack([g[:steps] for g in greedy]),
        relaxed=torch.stack(relaxed),
        relaxed_size=torch.stack(sizes),
        full_mlp=torch.tensor(full),
    )


def random_error(W, A, base, story_idx, emotion, k: int, sc: SparseConfig, generator) -> float:
    """Held-out error of random sets of size k, averaged over folds and draws."""
    errors = []
    for fold in range(sc.folds):
        split = fold_split(base, story_idx, fold)
        groups = emotion_groups(split, emotion)
        shift = activation_gap(A, groups.own, groups.other)
        target = heldout_target(W, A, split, emotion)
        errors += [set_error(W, shift, random_members(W.shape[1], k, generator), target) for _ in range(sc.random_sets)]
    return sum(errors) / len(errors)


def log_sizes(n: int) -> list[int]:
    return sorted({int(x) for x in torch.logspace(0, torch.log10(torch.tensor(float(n))), 40).round().tolist()})


def cv_summary(c: FoldCurves, ratios: list[float]) -> dict:
    mean_g, mean_r = c.greedy.mean(0), c.relaxed.mean(0)
    return {
        "full_mlp_error": c.full_mlp.tolist(),
        "greedy": {
            "k_star": int(mean_g.argmin()) + 1,
            "error_at_k_star": float(mean_g.min()),
            "curve": {str(k): float(mean_g[k - 1]) for k in log_sizes(len(mean_g))},
            "per_fold_k_star": (c.greedy.argmin(1) + 1).tolist(),
        },
        "relaxed": {
            "point_star": int(mean_r.argmin()),
            "lam_over_max_star": ratios[int(mean_r.argmin())],
            "error_at_star": float(mean_r.min()),
            "curve": [
                {"lam_over_max": r, "mean_size": float(c.relaxed_size[:, i].float().mean()), "error": float(mean_r[i])}
                for i, r in enumerate(ratios)
            ],  # fmt: skip
            "per_fold_size_at_star": c.relaxed_size[:, int(mean_r.argmin())].tolist(),
        },
    }


def members_of(order: Int[Tensor, " step"], k: int, n: int) -> torch.Tensor:
    members = torch.zeros(n, dtype=torch.bool)
    members[order[:k]] = True
    return members


def neuron_rows(W, d, shift, abar_own, abar_other, mask, relaxed_set, greedy: GreedyRun, greedy_set) -> list[dict]:
    """One row per neuron in either set. share: u_n . d / |d|^2, which sums to 1 over all neurons."""
    share = shift * (W.T @ d) / (d @ d)
    norm = shift * W.norm(dim=0) / d.norm()
    rank = {int(j): i + 1 for i, j in enumerate(greedy.order)}
    rows = [
        {"neuron": n, "in_relaxed_set": bool(relaxed_set[n]), "mask": round(float(mask[n]), 4),
         "in_greedy_set": bool(greedy_set[n]), "greedy_rank": rank.get(n, ""), "share_of_d": float(share[n]),
         "norm_over_d": float(norm[n].abs()), "shift": float(shift[n]), "abar_own": float(abar_own[n]),
         "abar_other": float(abar_other[n])}
        for n in (relaxed_set | greedy_set).nonzero().squeeze(1).tolist()
    ]  # fmt: skip
    return sorted(rows, key=lambda r: -r["share_of_d"])


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_emotion(W, WtW, A, base, story_idx, emotion: str, sc: SparseConfig, generator, t0) -> tuple[dict, list[dict], dict]:
    ratios = lambda_ratios(sc)
    cv = cv_summary(fold_curves(W, WtW, A, base, story_idx, emotion, sc, t0), ratios)
    k_star, point_star = cv["greedy"]["k_star"], cv["relaxed"]["point_star"]

    everything = no_heldout(base)
    final = run_searches(W, WtW, A, everything, emotion, sc)
    mask = final.path[point_star].mask
    relaxed_set = mask > 0
    greedy_set = members_of(final.greedy.order, k_star, W.shape[1])
    k_relaxed = int(relaxed_set.sum())
    cv["greedy"]["random_error_at_k_star"] = random_error(W, A, base, story_idx, emotion, k_star, sc, generator)
    cv["relaxed"]["random_error_at_size"] = random_error(W, A, base, story_idx, emotion, k_relaxed, sc, generator)

    summary = {
        "cv": cv,
        "final": {
            "relaxed": {
                "size": k_relaxed,
                "leftover": set_leftover(W, final.shift, relaxed_set),
                "fractional": int(((mask > 0) & (mask < 1)).sum()),
                "in_greedy_set": float((relaxed_set & greedy_set).sum() / k_relaxed),
                "greedy_leftover_at_same_size": float(final.greedy.leftover[min(k_relaxed, len(final.greedy.order)) - 1]),
            },
            "greedy": {
                "size": k_star,
                "leftover": float(final.greedy.leftover[k_star - 1]),
                "size_at_reference_leftover": first_below(final.greedy.leftover, sc.reference_leftover),
            },
        },
    }
    groups = emotion_groups(everything, emotion)
    abar_own, abar_other = A[groups.own].mean(0), A[groups.other].mean(0)
    d = W @ final.shift
    rows = neuron_rows(W, d, final.shift, abar_own, abar_other, mask, relaxed_set, final.greedy, greedy_set)
    tensors = {"d": d, "shift": final.shift, "mask": mask, "abar_own": abar_own, "abar_other": abar_other}
    return summary, rows, tensors


def report(emotion: str, s: dict, seconds: float) -> str:
    cv, f = s["cv"], s["final"]
    return (
        f"[{emotion}] relaxed: size {f['relaxed']['size']} (folds {cv['relaxed']['per_fold_size_at_star']}),"
        f" held-out error {cv['relaxed']['error_at_star']:.3f} (random {cv['relaxed']['random_error_at_size']:.3f}),"
        f" {f['relaxed']['in_greedy_set']:.0%} in greedy set"
        f" | greedy: size {f['greedy']['size']} (folds {cv['greedy']['per_fold_k_star']}), held-out error"
        f" {cv['greedy']['error_at_k_star']:.3f} (random {cv['greedy']['random_error_at_k_star']:.3f}),"
        f" size at leftover {f['greedy']['leftover']:.3f}"
        f" | all neurons {sum(cv['full_mlp_error']) / len(cv['full_mlp_error']):.3f} | {seconds:.0f}s"
    )


def main():
    cfg = load_run_config(sys.argv[1:])
    sc = cfg.sparse
    t0 = time.time()
    run_dir = cfg.runs_dir / sc.sums_run
    A = late_means(run_dir)["mlp_act"].float()
    records = load_story_records(run_dir)
    base = StorySplit.of(records)
    story_idx = torch.tensor([r.story_idx for r in records])
    W = load_w_down(cfg.model, cfg.layer).float()
    WtW = W.T @ W
    generator = torch.Generator().manual_seed(sc.seed)
    print(f"[load] A {tuple(A.shape)}, W {tuple(W.shape)}, {time.time() - t0:.0f}s", flush=True)

    summaries, tensors = {}, {}
    for emotion in sc.emotions:
        summary, rows, t = run_emotion(W, WtW, A, base, story_idx, emotion, sc, generator, t0)
        summaries[emotion], tensors[emotion] = summary, t
        write_csv(cfg.result_dir / "neurons" / f"{emotion}.csv", rows)
        print(report(emotion, summary, time.time() - t0), flush=True)

    out = {"model": cfg.model, "layer": cfg.layer, "config": sc.model_dump(), "emotions": summaries}
    (cfg.result_dir / "summary.json").write_text(json.dumps(out, indent=1))
    stacked = {k: torch.stack([tensors[e][k] for e in sc.emotions]).contiguous() for k in next(iter(tensors.values()))}
    save_file(stacked, str(cfg.result_dir / "tensors.safetensors"))
    print(f"[saved] {cfg.result_dir} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
