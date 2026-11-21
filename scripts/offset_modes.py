"""Are the offset (constant) neurons at their poles, or in between? CPU, from the signal_noise cache.

    python scripts/offset_modes.py --config configs/signal_noise.yaml

- Inputs: runs/signal_noise/story_classes.safetensors (per story: mean of a * 1[class] for high / low / middle, and
  the share of pole tokens), the p3 / p97 cache (`signal_noise.thresholds_file`), layer 25's W_down and the unit vectors.
- Groups as in `signal_noise.py`. Per group, on the emotion's train stories:
  - `pole_tokens`: share of tokens at a pole;
  - `push_from_poles`: share of the group's push (signed per neuron, so each neuron counts toward its own net push)
    that comes from pole tokens;
  - `middle_position`: the typical activation on middle tokens, placed on [p3, p97] (0 = p3, 1 = p97).
- And for the noise group: how much of its contrast comes from neurons with a moderate own-vs-other t (2 < |t| < 4.6).
- Writes results/signal_noise/offset_modes.json.
"""

import json
import statistics
import sys

import torch
from safetensors.torch import load_file

from emotion_neurons.config import load_run_config
from emotion_neurons.decomposition.signal_noise import classify
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.models import load_w_down
from emotion_neurons.texts import load_stories

torch.set_grad_enabled(False)
GROUPS = ("specific", "constant", "noise")


def group_profile(cm, pole_share, k, own, mask, net, p3, p97) -> dict:
    sign = net.sign()
    pushes = {c: (cm[c][own].mean(0) * k * sign)[mask] for c in ("high", "low", "middle")}
    total = sum(pushes.values()).sum()
    tokens_pole = pole_share[own][:, mask].mean(0)
    middle_value = cm["middle"][own][:, mask].mean(0) / (1 - tokens_pole).clamp_min(1e-6)
    position = (middle_value - p3[mask]) / (p97[mask] - p3[mask])
    return {
        "pole_tokens": float(tokens_pole.mean()),
        "push_from_poles": float((pushes["high"].sum() + pushes["low"].sum()) / total),
        "push_from_high": float(pushes["high"].sum() / total),
        "push_from_low": float(pushes["low"].sum() / total),
        "middle_position_median": float(position.median()),
        "middle_position_q25_q75": [float(x) for x in torch.quantile(position, torch.tensor([0.25, 0.75]))],
    }


def main():
    cfg = load_run_config(sys.argv[1:])
    sn = cfg.signal_noise
    t = load_file(str(cfg.runs_dir / cfg.run_id / "story_classes.safetensors"))
    cm = {c: t[c].float() for c in ("high", "low", "middle")}
    pole_share = t["pole_share"].float()
    q = load_file(str(sn.thresholds_file))
    p3, p97 = q["p3"].float(), q["p97"].float()
    W = load_w_down(cfg.model, cfg.layer).float()
    units = load_file(str(sn.layer_vectors_file))["unit"][cfg.layer].float()
    stories = load_stories(cfg.data)
    held = torch.tensor([s.heldout for s in stories])
    out = {}
    for e, name in enumerate(EMOTIONS):
        own_all = torch.tensor([s.emotion == name for s in stories])
        own, other = own_all & ~held, ~own_all & ~held
        k = W.T @ units[e]
        P = (cm["high"] + cm["low"] + cm["middle"]) * k
        g = classify(P[own], P[other], sn.t_cut)
        row = {grp: group_profile(cm, pole_share, k, own, getattr(g, grp), g.mean, p3, p97) for grp in GROUPS}
        contrast = g.mean - g.mean_other
        moderate = g.noise & (g.t_vs_other.abs() > 2)
        row["noise_contrast_from_moderate_t"] = float(contrast[moderate].sum() / contrast[g.noise].sum())
        row["noise_moderate_count"] = int(moderate.sum())
        row["noise_silent_on_own"] = float(
            contrast[g.noise & (g.mean_other.abs() > g.mean.abs())].sum() / contrast[g.noise].sum()
        )
        out[name] = row
        print(f"[{name}] " + " | ".join(f"{grp}: pole tokens {r['pole_tokens']:.3f}, push from poles {r['push_from_poles']:+.2f} "
              f"(high {r['push_from_high']:+.2f}, low {r['push_from_low']:+.2f}), middle at {r['middle_position_median']:.2f} of [p3, p97]"
              for grp, r in row.items() if grp in GROUPS)
              + f" | noise contrast from 2<|t|<4.6: {row['noise_contrast_from_moderate_t']:+.2f} ({row['noise_moderate_count']} neurons)", flush=True)  # fmt: skip
    mean = {grp: {key: statistics.mean(out[e][grp][key] for e in EMOTIONS) for key in ("pole_tokens", "push_from_poles", "push_from_high", "push_from_low", "middle_position_median")} for grp in GROUPS}  # fmt: skip
    mean["noise_contrast_from_moderate_t"] = statistics.mean(out[e]["noise_contrast_from_moderate_t"] for e in EMOTIONS)
    print("[mean] " + json.dumps(mean))
    path = cfg.results_dir / "signal_noise" / "offset_modes.json"
    path.write_text(json.dumps({"mean": mean, "emotions": out}, indent=1))
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
