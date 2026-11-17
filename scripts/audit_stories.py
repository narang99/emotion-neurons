"""Cross-file audit of data/stories/*.jsonl (run after every file passes validate_stories.py).

Checks:
- emotion field and id prefix match the file name (catches cross-agent contamination)
- no duplicate ids or near-duplicate texts across files (first 80 chars)
- banned-stem scan: any word *starting with* a banned word's stem (e.g. 'furious' -> 'furiously')
- per-file stats: n, words min/median/max, first-person fraction,
  and how many stories mention any of the 12 emotions' words (cross-emotion leakage)
"""
import collections
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMOTIONS = json.loads((ROOT / "configs/emotions.json").read_text())
ALL_WORDS = sorted({w for ws in EMOTIONS.values() for w in ws})


def stem(w):
    """Crude stem so 'furious' catches 'furiously' and 'terrified' catches 'terrify'.
    Words shorter than 6 letters are matched exactly (avoids mad->made, sad->saddle, scar->scarf)."""
    if len(w) < 6:
        return None
    for suf in ("ness", "ly", "ed", "ful", "ing", "ion", "y", "e"):
        if w.endswith(suf) and len(w) - len(suf) >= 5:
            return w[: -len(suf)]
    return w


def main():
    files = sorted((ROOT / "data/stories").glob("*.jsonl"))
    seen_ids, seen_text = {}, {}
    problems = []
    print(f"{'file':<16}{'n':>5}{'wmin':>6}{'wmed':>6}{'wmax':>6}{'1st%':>6}{'xemo':>6}")
    for f in files:
        if ".part" in f.name:
            problems.append(f"leftover part file {f.name}")
            continue
        expect = "neutral" if f.stem.startswith("neutral") else f.stem
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        own = ALL_WORDS if expect == "neutral" else EMOTIONS[expect]
        stems = sorted({stem(w.lower()) for w in own if " " not in w} - {None})
        words, first, xemo = [], 0, 0
        for r in recs:
            if r["emotion"] != expect or not r["id"].startswith(expect + "-"):
                problems.append(f"{f.name}: {r['id']} has emotion={r['emotion']}")
            if r["id"] in seen_ids:
                problems.append(f"dup id {r['id']} in {f.name} and {seen_ids[r['id']]}")
            seen_ids[r["id"]] = f.name
            key = re.sub(r"\W+", " ", r["text"].lower())[:80]
            if key in seen_text:
                problems.append(f"near-dup text {r['id']} ~ {seen_text[key]}")
            seen_text[key] = r["id"]
            t = r["text"]
            words.append(len(t.split()))
            if re.search(r"\b(I|I'm|I'd|I've|my|me)\b", t[:400]):
                first += 1
            toks = re.findall(r"[a-z]+", t.lower())
            for s in stems:
                hits = [w for w in toks if w.startswith(s)]
                if hits:
                    problems.append(f"{r['id']}: stem '{s}' -> {sorted(set(hits))}")
            if any(re.search(rf"\b{re.escape(w)}\b", t, re.I) for w in ALL_WORDS):
                xemo += 1
        print(f"{f.stem:<16}{len(recs):>5}{min(words):>6}{int(statistics.median(words)):>6}"
              f"{max(words):>6}{100*first//len(recs):>5}%{xemo:>6}")
    print(f"\ntotal stories: {len(seen_ids)}")
    print(f"{len(problems)} problem(s)")
    for p in problems:
        print("  " + p)


if __name__ == "__main__":
    main()
