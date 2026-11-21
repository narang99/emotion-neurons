"""Set size vs cosine with the vector being rebuilt, for the two set searches. CPU.

    uv run --with matplotlib python scripts/plot_set_curves.py --config configs/residual_sets.yaml --device cpu

- Method 1 (match sets, `sparse_set.py`): greedy on cosine; target = the denoised MLP write, 12-emotion-mean baseline.
  Curve: `results/sparse_l25_denoised/sparse_set.json`, method `greedy`. Dot: the match-set size.
- Method 2 (`residual_sets.py`): greedy on the leftover; target = W_down (own mean - other mean), raw activations.
  Curve: the cosine of each prefix with its target, recomputed here. Dot: the signal-set size (noise level).
- Each method is scored against its own target.
- Writes results/residual_sets/set_curves.png (log x) and set_curves_linear.png.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from emotion_neurons.config import load_run_config  # noqa: E402
from emotion_neurons.decomposition.residual import residual_greedy  # noqa: E402
from emotion_neurons.emotions import EMOTIONS  # noqa: E402
from emotion_neurons.models import load_w_down  # noqa: E402
from emotion_neurons.texts import load_stories  # noqa: E402

torch.set_grad_enabled(False)
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#262624", "#6b6a66", "#e6e5e1"


def prefix_cos(W, shift, order) -> torch.Tensor:
    d = W @ shift
    cos = []
    s = torch.zeros_like(d)
    for start in range(0, len(order), 500):  # in chunks, to keep memory small
        block = order[start : start + 500]
        cum = s[:, None] + (W[:, block] * shift[block]).cumsum(1)
        cos.append(torch.nn.functional.cosine_similarity(cum, d[:, None], dim=0))
        s = cum[:, -1]
    return torch.cat(cos)


def main():
    cfg = load_run_config(sys.argv[1:])
    method1 = json.loads((cfg.results_dir / "sparse_l25_denoised" / "sparse_set.json").read_text())["emotions"]
    method2 = json.loads((cfg.results_dir / "residual_sets" / "residual_sets.json").read_text())["emotions"]
    t = load_file(str(cfg.runs_dir / "signal_noise" / "story_classes.safetensors"))
    A = t["high"].float() + t["low"].float() + t["middle"].float()
    W = load_w_down(cfg.model, cfg.layer).float()
    WtW = W.T @ W
    stories = load_stories(cfg.data)
    held = torch.tensor([s.heldout for s in stories])

    curves = {}
    for name in EMOTIONS:
        own = torch.tensor([s.emotion == name for s in stories])
        gap = A[own & ~held].mean(0) - A[~own & ~held].mean(0)
        run = residual_greedy(W, WtW, gap, cfg.residual_sets.steps)
        g = method1[name]["methods"]["greedy"]
        curves[name] = (
            torch.tensor(g["cos"]),
            g["k_matches_full"],
            prefix_cos(W, gap, run.order),
            method2[name]["levels"]["noise"]["size"],
        )
    for scale, suffix in (("log", ""), ("linear", "_linear")):
        out = cfg.results_dir / "residual_sets" / f"set_curves{suffix}.png"
        draw(curves, scale).savefig(out, dpi=110, facecolor="#fcfcfb")
        print(f"[saved] {out}")


def draw(curves: dict, scale: str):
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
    for ax, (name, (c1, k1, c2, k2)) in zip(axes.flat, curves.items()):
        ax.plot(range(1, len(c1) + 1), c1, color=BLUE, lw=2, label="1: greedy on cosine (match set)")
        ax.plot(range(1, len(c2) + 1), c2, color=ORANGE, lw=2, label="2: greedy on leftover (signal set)")
        for k, c, color, offset, align in ((k1, c1, BLUE, (-8, 6), "right"), (k2, c2, ORANGE, (6, -14), "left")):
            ax.plot([k], [float(c[k - 1])], "o", ms=8, color=color, mec="white", mew=2, zorder=3)
            ax.annotate(f"{k}", (k, float(c[k - 1])), textcoords="offset points", xytext=offset, ha=align, fontsize=9, color=INK)
        ax.set_title(name, fontsize=11, color=INK, loc="left")
        ax.set_xscale(scale)
        ax.set_ylim(0, 1)
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    for ax in axes[-1]:
        ax.set_xlabel("neurons in the set" + (" (log scale)" if scale == "log" else ""), color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("cosine with the vector rebuilt", color=MUTED)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Layer-25 MLP emotion write: set size vs cosine (dot = where each method stops)", y=0.99, color=INK, fontsize=13)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


if __name__ == "__main__":
    main()
