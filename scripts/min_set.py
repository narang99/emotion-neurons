"""The smallest set of layer-L neurons that keeps the direction of each emotion's MLP write. CPU, no forward pass.

    python scripts/min_set.py --config configs/top64_l25.yaml

The write splits exactly over neurons: delta_e = sum_n u_n, with u_n = W_down[:, n] * dA_e[n] (dA = the neuron's
mean activation on emotion e's stories minus its mean over the 12 emotions). Keeping a set S means every other neuron
sits at its 12-emotion mean, so the write becomes sum_{n in S} u_n. Its direction is scored by cosine with delta_e.

Greedy forward selection: each step adds the neuron that raises cos(sum_S u_n, delta_e) the most. Also reported:
orders by projection c_n = u_n . delta_hat, by |u_n|, and random (the floor).

Selection for the mode analysis: round-robin over emotions down the greedy orders until `select` neurons.

Writes results/<run_id>/min_set.json (sizes per cosine threshold, greedy order and curve per emotion) and
results/<run_id>/selection.json ({"groups": {emotion: [neurons]}, "neurons": [...]}), read by harvest.py and modes.py
when harvest.neurons_file is set.
"""

import json
import sys

import torch
from safetensors.torch import load_file

from emotion_neurons.config import EMOTIONS, load_run_config
from emotion_neurons.weights import load_w_down

THRESHOLDS = (0.9, 0.95, 0.99)


def first_reaching(cos: torch.Tensor) -> list[int | None]:
    return [int((cos < t).sum()) + 1 if (cos >= t).any() else None for t in THRESHOLDS]


def cos_curve(U: torch.Tensor, d: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(U[:, order].cumsum(1), d[:, None], dim=0)


def greedy(U: torch.Tensor, d: torch.Tensor, steps: int) -> tuple[list[int], list[float]]:
    """Unit coefficients: a neuron is either kept as it is or held at its mean."""
    c = U.T @ (d / d.norm())
    nn2 = (U * U).sum(0)
    s = torch.zeros_like(d)
    sd = ss = 0.0
    used = torch.zeros(U.shape[1], dtype=torch.bool)
    order, cos = [], []
    for _ in range(steps):
        num = sd + c  # (s + u_n) . d_hat
        score = num / (ss + 2 * (U.T @ s) + nn2).clamp_min(1e-12).sqrt()
        score[used] = -9
        n = int(score.argmax())
        used[n] = True
        s = s + U[:, n]
        sd, ss = float(num[n]), float(s @ s)
        order.append(n)
        cos.append(sd / ss**0.5)
        if cos[-1] >= THRESHOLDS[-1]:
            break
    return order, cos


def main():
    cfg = load_run_config(sys.argv[1:])
    ms = cfg.min_set
    W = load_w_down(cfg.model, cfg.layer)
    vec = load_file(str(cfg.results_dir / cfg.neuron_set.vectors_run / "vectors.safetensors"))
    D, A = vec["mlp_out/mean/raw"].float(), vec["mlp_act/mean/raw"].float()
    gen = torch.Generator().manual_seed(0)
    out = {"thresholds": THRESHOLDS, "emotions": {}}
    print(f"{'emotion':10} rebuild | neurons for cos {THRESHOLDS}: greedy | by c_n | by |u_n| | random")
    for e, emo in enumerate(EMOTIONS):
        d, a = D[e], A[e]
        U = W * a[None, :]
        err = float((U.sum(1) - d).norm() / d.norm())
        c = U.T @ (d / d.norm())
        order, cos = greedy(U, d, ms.greedy_steps)
        r = {
            "rebuild_error": err,
            "greedy": first_reaching(torch.tensor(cos)),
            "by_c": first_reaching(cos_curve(U, d, c.argsort(descending=True))),
            "by_norm": first_reaching(cos_curve(U, d, U.norm(dim=0).argsort(descending=True))),
            "random": first_reaching(cos_curve(U, d, torch.randperm(U.shape[1], generator=gen))),
            "order": order,
            "cos": cos,
            "c_share": [float(c[n] / d.norm()) for n in order],
            "dA": [float(a[n]) for n in order],
        }
        out["emotions"][emo] = r
        at = {k: round(cos[k - 1], 2) for k in (8, 64, 256, len(cos)) if k <= len(cos)}
        print(f"{emo:10} {err:.4f}  | {r['greedy']} | {r['by_c']} | {r['by_norm']} | {r['random']} | greedy cos at {at}")

    # Round-robin down the greedy orders.
    sel, groups, rank = [], {emo: [] for emo in EMOTIONS}, 0
    while len(sel) < ms.select:
        for emo in EMOTIONS:
            n = out["emotions"][emo]["order"][rank]
            if n not in sel and len(sel) < ms.select:
                sel.append(n)
            if n in sel:
                groups[emo].append(n)
        rank += 1
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "min_set.json").write_text(json.dumps(out))
    (cfg.result_dir / "selection.json").write_text(json.dumps({"groups": groups, "neurons": sel, "depth": rank}, indent=1))
    print(f"[selection] {len(sel)} neurons, greedy ranks 1..{rank} per emotion")
    print(f"[saved] {cfg.result_dir}/min_set.json, selection.json")


if __name__ == "__main__":
    main()
