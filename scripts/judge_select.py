"""Pick neurons for blind judging: in the signal set but not the match set, in both, and random. CPU.

    python scripts/judge_select.py --config configs/judge_sets.yaml --device cpu

- Signal set (method 2): the residual greedy (`residual_sets.py`) at its noise-level size.
- Match set (method 1): `results/sparse_l25_denoised/sets.json`.
- Per emotion in `judge_sets.emotions`: `per_group` neurons only in the signal set, `per_group` in both;
  and `random_per_emotion` neurons in neither set.
- Each neuron gets a blind id. The key (id -> neuron, emotion, group) is written apart from what judges see.
- Writes results/<run_id>/selection.json.
"""

import json
import random
import sys

import torch
from safetensors.torch import load_file

from emotion_neurons.config import load_run_config
from emotion_neurons.decomposition.residual import residual_greedy
from emotion_neurons.models import load_w_down
from emotion_neurons.texts import load_stories

torch.set_grad_enabled(False)


def main():
    cfg = load_run_config(sys.argv[1:])
    js = cfg.judge_sets
    rng = random.Random(js.seed)
    t = load_file(str(cfg.runs_dir / "signal_noise" / "story_classes.safetensors"))
    A = t["high"].float() + t["low"].float() + t["middle"].float()
    W = load_w_down(cfg.model, cfg.layer).float()
    WtW = W.T @ W
    sizes = json.loads((cfg.results_dir / "residual_sets" / "residual_sets.json").read_text())["emotions"]
    match = json.loads((cfg.results_dir / "sparse_l25_denoised" / "sets.json").read_text())
    stories = load_stories(cfg.data)
    held = torch.tensor([s.heldout for s in stories])

    picks = []
    for name in js.emotions:
        own = torch.tensor([s.emotion == name for s in stories])
        gap = A[own & ~held].mean(0) - A[~own & ~held].mean(0)
        k = sizes[name]["levels"]["noise"]["size"]
        signal = set(residual_greedy(W, WtW, gap, k).order.tolist())
        M = set(match[name]["match"]["neurons"])
        neither = sorted(set(range(W.shape[1])) - signal - M)
        groups = {"only_signal": sorted(signal - M), "both": sorted(signal & M)}
        print(
            f"[{name}] signal {len(signal)}, match {len(M)}, only signal {len(groups['only_signal'])}, both {len(groups['both'])}, match only {len(M - signal)}"
        )
        for group, pool in groups.items():
            picks += [{"neuron": n, "emotion": name, "group": group} for n in rng.sample(pool, js.per_group)]
        picks += [{"neuron": n, "emotion": name, "group": "random"} for n in rng.sample(neither, js.random_per_emotion)]
    rng.shuffle(picks)
    for i, p in enumerate(picks):
        p["id"] = f"N{i:03d}"
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "selection.json").write_text(json.dumps(picks, indent=1))
    print(f"[saved] {cfg.result_dir}/selection.json, {len(picks)} neurons")


if __name__ == "__main__":
    main()
