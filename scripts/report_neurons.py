"""Per-neuron report for a min_set selection: why it was picked, how emotion-specific its activation shift is, where it
fires, and its Haiku-named modes. CPU.

    python scripts/report_neurons.py --config configs/top64_l25.yaml

Reads results/<run_id>/{selection,min_set,modes,harvest}.json, modes/names_*.jsonl, hits.safetensors,
runs/<run_id>/{window.npy,windows.json} and Stage 2's vectors. Writes results/<run_id>/neurons_report.md.
"""

import json
import sys
from collections import Counter

import numpy as np
from safetensors.torch import load_file

from emotion_neurons.config import load_run_config
from emotion_neurons.emotions import EMOTIONS


def main():
    cfg = load_run_config(sys.argv[1:])
    rd = cfg.result_dir
    sel = json.loads((rd / "selection.json").read_text())
    ms = json.loads((rd / "min_set.json").read_text())["emotions"]
    modes = json.loads((rd / "modes.json").read_text())["neurons"]
    names, summ = {}, {}
    for p in sorted((rd / "modes").glob("names_*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "summary" in r:
                    summ[int(r["neuron"])] = r["summary"]
                else:
                    names[(int(r["neuron"]), str(r["cluster"]))] = r
    A = load_file(str(cfg.results_dir / cfg.neuron_set.vectors_run / "vectors.safetensors"))["mlp_act/mean/raw"].float()
    hits = {int(k): v.numpy() for k, v in load_file(str(rd / "hits.safetensors")).items()}
    win = np.load(cfg.run_dir / "window.npy")
    corpus = np.array([c for c, _ in json.loads((cfg.run_dir / "windows.json").read_text())])

    lines = ["# What the top 64 neurons of the emotion writes do (layer 25)", "",
             "Picked round-robin down each emotion's greedy order (the neurons that best keep the direction of the MLP's",
             "emotion write). Per neuron: the emotions it was picked for (greedy rank, share of |delta_e| it carries along",
             "delta_e); its activation shift dA over the 12 emotions (the top 3, in units of its largest |dA|); the share of",
             "its p97+ hits from TinyStories / FineWeb-edu / UltraChat; Haiku's one-line summary; its modes (share of hits,",
             "emotion label).", ""]  # fmt: skip
    for n in sel["neurons"]:
        picked = [f"{e} #{ms[e]['order'].index(n) + 1} ({ms[e]['c_share'][ms[e]['order'].index(n)]:+.3f})"
                  for e in EMOTIONS if n in sel["groups"].get(e, [])]  # fmt: skip
        a = A[:, n]
        top = a.abs().argsort(descending=True)[:3].tolist()
        shift = ", ".join(f"{EMOTIONS[i]} {float(a[i] / a.abs().max()):+.2f}" for i in top)
        cc = Counter(corpus[win[hits[n]]].tolist())
        tot = sum(cc.values())
        where = " / ".join(f"{cc.get(c, 0) / tot:.0%}" for c in ("tinystories", "fineweb-edu", "ultrachat"))
        lines.append(f"## {n}: {', '.join(picked)}")
        lines.append(f"- dA: {shift} | hits TS/FW/UC: {where}")
        lines.append(f"- {summ.get(n, '(no summary)')}")
        m = modes[str(n)]
        for c in sorted(m["clusters"], key=lambda c: -c["with_assigned"]):
            nm = names.get((n, str(c["id"])), {})
            lines.append(f"  - {c['with_assigned'] / m['n_hits']:.0%} [{nm.get('emotion', '?')}] {nm.get('mode', '?')}")
        nm = names.get((n, "other"), {})
        lines.append(f"  - {m['other'] / m['n_hits']:.0%} unclustered: {nm.get('mode', '-')}")
        lines.append("")
    (rd / "neurons_report.md").write_text("\n".join(lines))
    print(f"[saved] {rd}/neurons_report.md ({len(sel['neurons'])} neurons)")


if __name__ == "__main__":
    main()
