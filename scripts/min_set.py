"""The smallest set of layer-L neurons that keeps the direction of each emotion's MLP write. CPU, no forward pass.

    python scripts/min_set.py --config configs/top64_l25.yaml

- Greedy on the neuron terms (`emotion_neurons.decomposition.min_set`), against orders by projection c_n,
  by |u_n| and random.
- Selection for the mode analysis: round-robin over emotions down the greedy orders until `select` neurons.
- Writes results/<run_id>/min_set.json and selection.json ({"groups": {emotion: [...]}, "neurons": [...], "depth"}).
"""

import json
import sys

import torch

from emotion_neurons.config import load_run_config
from emotion_neurons.decomposition import neuron_terms
from emotion_neurons.decomposition.min_set import THRESHOLDS, first_reaching, greedy_on_terms, prefix_cos, round_robin
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.models import load_w_down
from emotion_neurons.vectors import EmotionVectors


def main():
    cfg = load_run_config(sys.argv[1:])
    W = load_w_down(cfg.model, cfg.layer)
    vectors = EmotionVectors.load(cfg.results_dir / cfg.neuron_set.vectors_run / "vectors.safetensors")
    write, shift = vectors.mlp_write(), vectors.activation_shift()
    generator = torch.Generator().manual_seed(0)
    out = {"thresholds": THRESHOLDS, "emotions": {}}
    print(f"{'emotion':10} rebuild | neurons for cos {THRESHOLDS}: greedy | by c_n | by |u_n| | random")
    for e, emotion in enumerate(EMOTIONS):
        d, a = write[e], shift[e]
        U = neuron_terms(W, a)
        err = float((U.sum(1) - d).norm() / d.norm())
        c = U.T @ (d / d.norm())
        order, cos = greedy_on_terms(U, d, cfg.min_set.greedy_steps)
        r = {
            "rebuild_error": err,
            "greedy": first_reaching(torch.tensor(cos)),
            "by_c": first_reaching(prefix_cos(U, d, c.argsort(descending=True))),
            "by_norm": first_reaching(prefix_cos(U, d, U.norm(dim=0).argsort(descending=True))),
            "random": first_reaching(prefix_cos(U, d, torch.randperm(U.shape[1], generator=generator))),
            "order": order,
            "cos": cos,
            "c_share": [float(c[n] / d.norm()) for n in order],
            "dA": [float(a[n]) for n in order],
        }
        out["emotions"][emotion] = r
        at = {k: round(cos[k - 1], 2) for k in (8, 64, 256, len(cos)) if k <= len(cos)}
        print(f"{emotion:10} {err:.4f}  | {r['greedy']} | {r['by_c']} | {r['by_norm']} | {r['random']} | greedy cos at {at}")

    orders = {emotion: out["emotions"][emotion]["order"] for emotion in EMOTIONS}
    selected, groups, depth = round_robin(orders, cfg.min_set.select)
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "min_set.json").write_text(json.dumps(out))
    (cfg.result_dir / "selection.json").write_text(json.dumps({"groups": groups, "neurons": selected, "depth": depth}, indent=1))
    print(f"[selection] {len(selected)} neurons, greedy ranks 1..{depth} per emotion")
    print(f"[saved] {cfg.result_dir}/min_set.json, selection.json")


if __name__ == "__main__":
    main()
