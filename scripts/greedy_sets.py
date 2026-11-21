"""Greedy sets: the fewest neurons of one MLP layer that rebuild 90% of each emotion's write. CPU.

    python scripts/greedy_sets.py --config configs/greedy_sets_l25.yaml

- Write per emotion: `d = W_down @ (abar_emotion - abar_all)`, story means of `mlp_act` over tokens 50+.
  - abar_emotion: the emotion's 120 stories. abar_all: all 1680 stories. No held-out split, no PCA.
- Greedy (`emotion_neurons.sparse`): add the neuron that rebuilds the most of what is still missing.
  - Neurons outside the set sit at their all-story mean.
  - Stop when the set rebuilds `greedy_sets.rebuilt` (90%) of d.
- Reads runs/<greedy_sets.sums_run>/{sums.safetensors, stories.json} and the layer's W_down.
- Writes results/<run_id>/:
  - sets.json: per emotion, the set's neurons in the order greedy added them;
  - neurons/<emotion>.csv: one row per neuron in the set;
  - summary.json: per emotion, the set size, and the size at other shares;
  - tensors.safetensors: `write`, `shift`, `abar_emotion` [12, dim] and `abar_all` [dim], rows in EMOTIONS order.
"""

import csv
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from emotion_neurons.capture import late_means
from emotion_neurons.config import load_run_config
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.models import load_w_down
from emotion_neurons.sparse import GreedyRun, emotion_shift, first_reaching, greedy_set
from emotion_neurons.texts import StorySplit, load_story_records

torch.set_grad_enabled(False)
REPORTED_SHARES = (0.5, 0.75, 0.9, 0.95)


def neuron_rows(W, shift, abar_emotion, abar_all, run: GreedyRun, size: int) -> list[dict]:
    """write_share_pct: the neuron's part of the write, along the write. Over all 12288 neurons it sums to 100."""
    d = W @ shift
    share = 100 * shift * (W.T @ d) / (d @ d)
    return [
        {"rank": i + 1, "neuron": n, "write_share_pct": round(float(share[n]), 4),
         "rebuilt_after_pct": round(100 * float(run.rebuilt[i]), 2), "shift": float(shift[n]),
         "abar_emotion": float(abar_emotion[n]), "abar_all": float(abar_all[n])}
        for i, n in enumerate(run.order[:size].tolist())
    ]  # fmt: skip


def summarise(run: GreedyRun, size: int) -> dict:
    return {
        "size": size,
        "rebuilt_pct": round(100 * float(run.rebuilt[size - 1]), 2),
        "size_at_pct": {str(round(100 * s)): first_reaching(run.rebuilt, s) for s in REPORTED_SHARES},
        "greedy_stopped_at": {"size": len(run.order), "rebuilt_pct": round(100 * float(run.rebuilt[-1]), 2)},
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    cfg = load_run_config(sys.argv[1:])
    gc = cfg.greedy_sets
    t0 = time.time()
    run_dir = cfg.runs_dir / gc.sums_run
    A = late_means(run_dir)["mlp_act"].float()
    split = StorySplit.of(load_story_records(run_dir))
    W = load_w_down(cfg.model, cfg.layer).float()
    WtW = W.T @ W
    abar_all = A.mean(0)
    print(f"[load] A {tuple(A.shape)}, W {tuple(W.shape)}, {time.time() - t0:.0f}s", flush=True)

    sets, summary, shifts = {}, {}, []
    for e, emotion in enumerate(EMOTIONS):
        own = split.label == e
        shift = emotion_shift(A, own)
        run = greedy_set(W, WtW, shift, steps=W.shape[1])
        size = first_reaching(run.rebuilt, gc.rebuilt)
        assert size is not None, f"{emotion}: greedy stopped at {float(run.rebuilt[-1]):.3f} before {gc.rebuilt}"
        sets[emotion] = run.order[:size].tolist()
        summary[emotion] = summarise(run, size)
        shifts.append(shift)
        write_csv(cfg.result_dir / "neurons" / f"{emotion}.csv", neuron_rows(W, shift, A[own].mean(0), abar_all, run, size))
        s = summary[emotion]
        print(f"[{emotion}] {size} neurons rebuild {s['rebuilt_pct']}% | sizes at 50/75/90/95%: {list(s['size_at_pct'].values())}"
              f" | greedy stopped at {s['greedy_stopped_at']['size']} ({s['greedy_stopped_at']['rebuilt_pct']}%) | {time.time() - t0:.0f}s", flush=True)  # fmt: skip

    meta = {"model": cfg.model, "layer": cfg.layer, "config": gc.model_dump(), "emotions": list(EMOTIONS)}
    (cfg.result_dir / "sets.json").write_text(json.dumps({**meta, "sets": sets}, indent=1))
    (cfg.result_dir / "summary.json").write_text(json.dumps({**meta, "summary": summary}, indent=1))
    shift = torch.stack(shifts)
    save_file(
        {"write": (shift @ W.T).contiguous(), "shift": shift, "abar_emotion": shift + abar_all, "abar_all": abar_all},
        str(cfg.result_dir / "tensors.safetensors"),
    )
    print(f"[saved] {cfg.result_dir} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
