"""Signal set and cancelling rest: the fewest neurons that rebuild an emotion's MLP write in direction and length. CPU.

    python scripts/residual_sets.py --config configs/residual_sets.yaml --device cpu

- Target per emotion: `d = W_down (abar_own - abar_other)`, raw story-mean activations of layer 25, train stories.
  `other` = the 11 other emotions and neutral.
- Greedy on the leftover (`decomposition.residual`): the set S is read where `|d - sum_S u_n| / |d|` first falls
  below each level, and below `noise`: the train gap's own sampling noise (rebuilding past it fits noise).
  The neurons outside S are the cancelling rest.
- Held-out checks per level:
  - the leftover on the held-out gap, and S's cosine with it;
  - the leftover against the noise floor: what the held-out leftover would be if the rest summed to exactly zero
    (from the spread of the rest's per-story sums: `tr cov / n` for own and other stories).
- Controls: random sets of the same size; the match sets, scored by leftover.
- Poles: the share of S's write (along itself) from tokens >= p97, <= p3 and middle; S's share of pole tokens.
- Inputs: the signal_noise cache (`runs/signal_noise/story_classes.safetensors`) and layer 25's W_down.
- Writes results/<run_id>/residual_sets.json.
"""

import json
import random
import statistics
import sys
import time

import torch
from safetensors.torch import load_file

from emotion_neurons.config import load_run_config
from emotion_neurons.decomposition.residual import first_below, residual_greedy
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.models import load_w_down
from emotion_neurons.texts import load_stories

torch.set_grad_enabled(False)
CLASSES = ("high", "low", "middle")


def leftover(W, shift, members) -> float:
    d = W @ shift
    return float((d - W[:, members] @ shift[members]).norm() / d.norm())


def noise_floor(W, acts_own, acts_other, rest) -> float:
    """Expected |gap of the rest's sums| if the rest summed to zero on average: from the spread over stories."""

    def spread(acts):
        x = acts[:, rest] @ W[:, rest].T
        return float(x.var(0).sum()) / len(x)

    return (spread(acts_own) + spread(acts_other)) ** 0.5


def heldout_checks(W, A, own_h, other_h, members) -> dict:
    gap = A[own_h].mean(0) - A[other_h].mean(0)
    d = W @ gap
    s = W[:, members] @ gap[members]
    rest = torch.ones(W.shape[1], dtype=torch.bool)
    rest[members] = False
    left = float((d - s).norm())
    if not rest.any():  # all neurons: report the noise of the whole held-out gap instead
        floor = noise_floor(W, A[own_h], A[other_h], ~rest)
        return {"leftover": 0.0, "cos": 1.0, "leftover_over_noise_floor": None, "noise_floor_share": floor / float(d.norm())}
    floor = noise_floor(W, A[own_h], A[other_h], rest)
    return {"leftover": left / float(d.norm()), "cos": float(torch.nn.functional.cosine_similarity(s, d, dim=0)),
            "leftover_over_noise_floor": left / floor, "noise_floor_share": floor / float(d.norm())}  # fmt: skip


def pole_split(W, cm, own, other, members, pole_share) -> dict:
    parts = {c: W[:, members] @ (cm[c][own].mean(0) - cm[c][other].mean(0))[members] for c in CLASSES}
    s = sum(parts.values())
    out = {c: float(v @ s / (s @ s)) for c, v in parts.items()}
    out["pole_tokens"] = float(pole_share[own][:, members].mean())
    return out


def main():
    cfg = load_run_config(sys.argv[1:])
    rs = cfg.residual_sets
    t0 = time.time()
    t = load_file(str(cfg.runs_dir / "signal_noise" / "story_classes.safetensors"))
    cm = {c: t[c].float() for c in CLASSES}
    A = cm["high"] + cm["low"] + cm["middle"]
    pole_share = t["pole_share"].float()
    W = load_w_down(cfg.model, cfg.layer).float()
    WtW = W.T @ W
    sets = {e: torch.tensor(v["match"]["neurons"]) for e, v in json.loads(cfg.signal_noise.sets_file.read_text()).items()}
    stories = load_stories(cfg.data)
    held = torch.tensor([s.heldout for s in stories])
    rng = random.Random(rs.seed)
    print(f"[load] {len(stories)} stories, WtW {tuple(WtW.shape)}, {time.time() - t0:.0f}s", flush=True)

    out = {"levels": rs.levels, "emotions": {}}
    for name in EMOTIONS:
        is_own = torch.tensor([s.emotion == name for s in stories])
        own, other, own_h, other_h = is_own & ~held, ~is_own & ~held, is_own & held, ~is_own & held
        gap = A[own].mean(0) - A[other].mean(0)
        run = residual_greedy(W, WtW, gap, rs.steps)
        row = {"min_leftover": float(run.leftover.min()), "steps_run": len(run.order),
               "at_size": {str(k): float(run.leftover[k - 1]) for k in rs.sizes if k <= len(run.order)}, "levels": {}}  # fmt: skip
        M = sets[name]
        row["match_set"] = {"size": len(M), "leftover": leftover(W, gap, M), "heldout": heldout_checks(W, A, own_h, other_h, M)}
        row["all_neurons_heldout"] = heldout_checks(W, A, own_h, other_h, torch.arange(W.shape[1]))
        everyone = torch.ones(W.shape[1], dtype=torch.bool)
        train_noise = noise_floor(W, A[own], A[other], everyone) / float((W @ gap).norm())
        row["train_gap_noise"] = train_noise
        for level in [*rs.levels, train_noise]:
            k = first_below(run.leftover, level)
            if k is None:
                row["levels"]["noise" if level == train_noise else str(level)] = None
                continue
            S = run.order[:k]
            randoms = [torch.tensor(rng.sample(range(W.shape[1]), k)) for _ in range(rs.random_sets)]
            row["levels"]["noise" if level == train_noise else str(level)] = {
                "size": k,
                "heldout": heldout_checks(W, A, own_h, other_h, S),
                "random_leftover": statistics.mean(leftover(W, gap, R) for R in randoms),
                "random_heldout_leftover": statistics.mean(heldout_checks(W, A, own_h, other_h, R)["leftover"] for R in randoms),
                "poles": pole_split(W, cm, own, other, S, pole_share),
                "in_match_set": float(torch.isin(S, M).float().mean()),
            }  # fmt: skip
        out["emotions"][name] = row
        line = f"[{name}] min leftover {row['min_leftover']:.3f} after {row['steps_run']} | match set {len(M)}: leftover {row['match_set']['leftover']:.2f}"
        for level, v in row["levels"].items():
            if v:
                h = v["heldout"]
                line += (f" | <= {level}: k {v['size']} (random {v['random_leftover']:.2f}), held-out leftover {h['leftover']:.2f}"
                         f" = {h['leftover_over_noise_floor']:.2f} x noise floor, cos {h['cos']:.2f}, poles {v['poles']['high'] + v['poles']['low']:.2f}")  # fmt: skip
        a = row["all_neurons_heldout"]
        print(line + f" | held-out gap noise {a['noise_floor_share']:.2f} of its length | {time.time() - t0:.0f}s", flush=True)

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "residual_sets.json").write_text(json.dumps(out, indent=1))
    print(f"[saved] {cfg.result_dir}/residual_sets.json")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
