"""Unblind and summarise the Haiku judgments of `judge_snippets.py`. CPU.

    python scripts/judge_aggregate.py --config configs/judge_sets.yaml --device cpu

- Reads results/<run_id>/selection.json (the key) and results/<run_id>/judgments/*.jsonl.
- Per group (only in the signal set, in both sets, random), over all emotions and per emotion:
  the share judged related to the named emotion (score >= 1, and = 2, with 95% Wilson intervals), the share related
  to any emotion, and the share of each kind.
- Writes results/<run_id>/summary.json.
"""

import json
import math
import sys
from collections import Counter

from emotion_neurons.config import load_run_config

GROUPS = ("both", "only_signal", "random")
KINDS = ("word", "situation", "body", "syntax", "topic", "unclear")


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    related = sum(r["emotion_named"] >= 1 for r in rows)
    clear = sum(r["emotion_named"] == 2 for r in rows)
    kinds = Counter(r["kind"] for r in rows)
    return {
        "n": n,
        "named_related": related / n, "named_related_ci": wilson(related, n),
        "named_clear": clear / n, "named_clear_ci": wilson(clear, n),
        "any_emotion": sum(r["any_emotion"] >= 1 for r in rows) / n,
        "kinds": {k: kinds.get(k, 0) / n for k in KINDS},
    }  # fmt: skip


def main():
    cfg = load_run_config(sys.argv[1:])
    key = {p["id"]: p for p in json.loads((cfg.result_dir / "selection.json").read_text())}
    judged = {}
    for f in sorted((cfg.result_dir / "judgments").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                judged[r["id"]] = r | key[r["id"]]
    missing = sorted(set(key) - set(judged))
    print(f"[judged] {len(judged)} of {len(key)}; missing {missing}")
    rows = list(judged.values())
    out = {"overall": {g: summarise([r for r in rows if r["group"] == g]) for g in GROUPS}, "per_emotion": {}}
    for e in sorted({r["emotion"] for r in rows}):
        out["per_emotion"][e] = {g: summarise([r for r in rows if r["group"] == g and r["emotion"] == e]) for g in GROUPS}
    out["examples"] = {
        g: [{"neuron": r["neuron"], "emotion": r["emotion"], "theme": r["theme"], "named": r["emotion_named"], "kind": r["kind"]}
            for r in rows if r["group"] == g][:12]
        for g in GROUPS
    }  # fmt: skip
    for g, s in out["overall"].items():
        print(f"[{g}] n {s['n']} | related to the named emotion {s['named_related']:.2f} {[round(x, 2) for x in s['named_related_ci']]}"
              f" | clearly {s['named_clear']:.2f} | any emotion {s['any_emotion']:.2f} | kinds " + ", ".join(f"{k} {v:.2f}" for k, v in s["kinds"].items()))  # fmt: skip
    (cfg.result_dir / "summary.json").write_text(json.dumps(out, indent=1))
    print(f"[saved] {cfg.result_dir}/summary.json")


if __name__ == "__main__":
    main()
