"""Neuron classes under a second scene (e.g. harm), next to the report's own classes. CPU.

    python scripts/scene_classes.py --config configs/harvest_refusal_qwen1_bands.yaml --judge.scene harm

- Needs the report's cards.json and judgements_<scene>/ (`scripts/judge_modes.py --judge.scene <scene>`).
- Writes results/<run_id>/classes_<scene>.csv and prints counts and write shares per class.
"""

import csv
import json
import sys
from collections import defaultdict

from emotion_neurons.config import ModesRunConfig, load_run_config
from emotion_neurons.judge import judged_scene, judgements_dir
from emotion_neurons.modes_report import read_judgements, scene_classes

CLASSES = ("match", "split", "no match")


def table(rows: list[dict], key: str) -> str:
    count, share = defaultdict(int), defaultdict(float)
    for r in rows:
        count[r[key]] += 1
        share[r[key]] += r["share"]
    return " | ".join(f"{c} {count[c]} ({share[c]:.1f}% of the write)" for c in CLASSES)


if __name__ == "__main__":
    cfg = load_run_config(ModesRunConfig, sys.argv[1:])
    scene = judged_scene(cfg)
    cards = json.loads((cfg.result_dir / "cards.json").read_text())
    rows = scene_classes(cards, read_judgements(judgements_dir(cfg)))
    neurons = [r for r in rows if not r["is_control"]]
    top = [r for r in neurons if r["rank"] <= 40]
    print(f"[{cfg.concept}] set: {table(neurons, 'own_class')}")
    print(f"[{scene}] set: {table(neurons, 'class')}")
    print(f"[{scene}] top 40: {table(top, 'class')}")
    print(f"[{scene}] controls: {table([r for r in rows if r['is_control']], 'class')}")
    cross = defaultdict(int)
    for r in neurons:
        cross[(r["own_class"], r["class"])] += 1
    print(f"[cross] ({cfg.concept}, {scene}): {dict(cross)}")
    out = cfg.result_dir / f"classes_{scene}.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {out}")
