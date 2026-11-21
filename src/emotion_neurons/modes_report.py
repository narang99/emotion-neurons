"""Data for the modes report: one card per harvested neuron, with both poles, judge verdicts and the logit lens.

- The happy side of a neuron is the pole happy moves it toward: high if its happy shift is positive, else low.
- Neuron class, from the clusters on its happy side (a cluster is happy for a judge at 3+ of 5 windows):
  - "happy": at least one cluster is happy for every judge that judged it;
  - "split": no such cluster, but at least one is happy for one judge;
  - "not happy": no cluster on the happy side is happy for either judge.
"""

import json
import re
from pathlib import Path

HAPPY_AT = 3


def read_cluster_texts(modes_dir: Path) -> dict[tuple, list]:
    """{(neuron, pole, cluster id or "other"): [[activation, window], ...]} from the modes batch files."""
    texts: dict[tuple, list] = {}
    neuron = pole = group = None
    for path in sorted(modes_dir.glob("batch_*.txt")):
        for line in path.read_text().splitlines():
            if m := re.match(r"##### neuron (\d+)", line):
                neuron, pole, group = int(m.group(1)), None, None
            elif m := re.match(r"=== (high|low) pole", line):
                pole, group = m.group(1), None
            elif m := re.match(r"--- cluster (\d+)", line):
                group = int(m.group(1))
            elif line.startswith("--- unclustered"):
                group = "other"
            elif group is not None and (m := re.match(r"\s*\[([-\d.]+)\]\s?(.*)", line)):
                texts.setdefault((neuron, pole, group), []).append([float(m.group(1)), m.group(2)])
    return texts


def read_judgements(judgements_dir: Path) -> dict[str, dict[tuple, dict]]:
    """{model folder name: {(neuron, pole, cluster): verdict}}."""
    out = {}
    for folder in sorted(p for p in judgements_dir.iterdir() if p.is_dir()):
        verdicts = {}
        for path in sorted(folder.glob("batch_*.jsonl")):
            for line in path.read_text().splitlines():
                v = json.loads(line)
                verdicts[(v["neuron"], v["pole"], v["cluster"])] = v
        out[folder.name] = verdicts
    return out


def happy_side(shift: float) -> str:
    return "high" if shift >= 0 else "low"


def neuron_class(clusters: list[dict], judges: list[str]) -> str:
    """A judge with no verdict on a cluster casts no vote there (Gemini skips 55 of 4483 clusters every time)."""
    votes = [[c["verdicts"][j]["happy_windows"] >= HAPPY_AT for j in judges if j in c["verdicts"]] for c in clusters]
    if any(v and all(v) for v in votes):
        return "happy"
    if any(any(v) for v in votes):
        return "split"
    return "not happy"


def neuron_card(n: int, info: dict, modes: dict, texts: dict, judgements: dict, lens: dict) -> dict:
    """info: rank, share, shift, cos, is_control. modes: this neuron's entry of modes.json."""
    judges = list(judgements)
    poles = {}
    for pole in ("high", "low"):
        clusters = []
        for c in modes[pole]["clusters"]:
            key = (n, pole, c["id"])
            clusters.append({
                "id": c["id"], "size": c["with_assigned"], "mean_act": round(c["mean_act"], 3),
                "examples": texts.get(key, []),
                "verdicts": {j: judgements[j][key] for j in judges if key in judgements[j]},
            })  # fmt: skip
        poles[pole] = {"n_hits": modes[pole]["n_hits"], "other": modes[pole]["other"], "clusters": clusters,
                       "other_examples": texts.get((n, pole, "other"), [])}  # fmt: skip
    side = happy_side(info["shift"])
    return {
        "neuron": n, **info, "happy_side": side, "class": neuron_class(poles[side]["clusters"], judges),
        "lens": lens, "poles": poles,
    }  # fmt: skip
