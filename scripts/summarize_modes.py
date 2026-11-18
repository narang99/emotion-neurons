"""Stage 4b summary: join Haiku's mode names (results/<run_id>/modes/names_XX.jsonl) with modes.json and the groups.

    python scripts/summarize_modes.py --config configs/stage3_l25.yaml

Writes results/<run_id>/modes_summary.md: per group (emotion/pole, or emotion for a min_set selection), the neurons,
each with Haiku's one-line summary (if written) and its modes (cluster size share, mean activation, name, related emotion), and how often a mode is
named as related to the neuron's own emotion, against the random controls.
"""

import json
import sys
from collections import defaultdict

from emotion_neurons.config import load_run_config


def main():
    cfg = load_run_config(sys.argv[1:])
    rd = cfg.result_dir
    modes = json.loads((rd / "modes.json").read_text())
    names, summaries = {}, {}
    for p in sorted((rd / "modes").glob("names_*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "summary" in r:
                    summaries[int(r["neuron"])] = r["summary"]
                else:
                    names[(int(r["neuron"]), str(r["cluster"]))] = r
    groups = modes["groups"]
    lines = ["# Modes of the layer-25 neurons behind the MLP's emotion write", "",
             "Per (emotion, pole): pos = pushes the kept dims along the write, neg = against it. Each mode is an HDBSCAN",
             "cluster of the neuron's p97+ hits (share = hits in it, incl. assigned noise), named by Haiku from 5 exemplars.", ""]  # fmt: skip
    stats = defaultdict(lambda: [0, 0])  # group kind -> [modes naming own emotion, modes]
    missing = 0
    for g, ns in groups.items():
        emo = g.split("/")[0]
        lines.append(f"## {g}")
        for n in ns:
            r = modes["neurons"][str(n)]
            parts = []
            for c in r["clusters"] + [{"id": "other", "with_assigned": r["other"]}]:
                nm = names.get((n, str(c["id"])))
                if nm is None:
                    missing += c["id"] != "other"
                    continue
                share = c["with_assigned"] / r["n_hits"]
                parts.append(f"{nm['mode']} ({share:.0%}, {nm.get('emotion', 'none')})")
                kind = "control" if g == "control" else (g.split("/")[1] if "/" in g else "selected")
                if c["id"] != "other":
                    target = emo if g != "control" else None
                    stats[kind][1] += 1
                    stats[kind][0] += nm.get("emotion") == target if target else nm.get("emotion", "none") != "none"
            head = f"- **{n}**" + (f" — {summaries[n]}" if n in summaries else "")
            lines.append(head)
            lines.extend(f"  - {p}" for p in parts) if parts else lines.append("  - (no named modes)")
        lines.append("")
    lines.insert(5, "Share of clustered modes named as the neuron's own emotion (controls: as any emotion): "
                 + ", ".join(f"{k} {a}/{b}" for k, (a, b) in stats.items()) + "\n")  # fmt: skip
    (rd / "modes_summary.md").write_text("\n".join(lines))
    print(f"[summary] {len(names)} names joined, {missing} clusters without a name; stats {dict(stats)}")
    print(f"[saved] {rd}/modes_summary.md")


if __name__ == "__main__":
    main()
