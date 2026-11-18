"""Validate a stories JSONL file.

Usage: python scripts/validate_stories.py data/stories/<name>.jsonl [--per-topic N]

Each line: {"id": str, "emotion": str ("neutral" allowed), "topic_id": int, "story_idx": int, "text": str}
Checks: valid JSON, required keys, unique ids, topic_id in range, per-topic counts,
word count 100-260, and no banned emotion words (for neutral: no word from any emotion's list).
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOPICS = json.loads((ROOT / "configs/topics.json").read_text())
EMOTIONS = json.loads((ROOT / "configs/emotions.json").read_text())


def banned_for(emotion):
    if emotion == "neutral":
        return sorted({w for ws in EMOTIONS.values() for w in ws})
    return EMOTIONS[emotion]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--per-topic", type=int, default=None)
    args = ap.parse_args()

    errors, ids, counts = [], set(), collections.Counter()
    lines = Path(args.path).read_text().splitlines()
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {i}: bad JSON ({e})")
            continue
        missing = {"id", "emotion", "topic_id", "story_idx", "text"} - r.keys()
        if missing:
            errors.append(f"line {i}: missing {missing}")
            continue
        if r["id"] in ids:
            errors.append(f"line {i}: duplicate id {r['id']}")
        ids.add(r["id"])
        if r["emotion"] != "neutral" and r["emotion"] not in EMOTIONS:
            errors.append(f"line {i}: unknown emotion {r['emotion']}")
            continue
        if not (0 <= r["topic_id"] < len(TOPICS)):
            errors.append(f"line {i}: topic_id out of range")
        counts[r["topic_id"]] += 1
        n_words = len(r["text"].split())
        if not (100 <= n_words <= 260):
            errors.append(f"line {i} ({r['id']}): {n_words} words, want 100-260")
        for w in banned_for(r["emotion"]):
            if re.search(rf"\b{re.escape(w)}\b", r["text"], re.IGNORECASE):
                errors.append(f"line {i} ({r['id']}): banned word '{w}'")

    if args.per_topic is not None:
        for t in range(len(TOPICS)):
            if counts[t] != args.per_topic:
                errors.append(f"topic {t}: {counts[t]} stories, want {args.per_topic}")

    print(f"{len(ids)} stories, per-topic counts: {dict(sorted(counts.items()))}")
    if errors:
        print(f"{len(errors)} problem(s):")
        print("\n".join(errors))
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
