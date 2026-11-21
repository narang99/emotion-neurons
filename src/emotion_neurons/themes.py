"""Themes: what kinds of meaning club together to build an emotion's direction, read off the cluster labels.

1. Lines: one per cluster on the side the emotion uses, for the set's neurons (controls left out). A line holds the
   cluster id, the neuron's lens on that side, the first judge's mode name and scene, and one short exemplar.
   Lines are ordered by weight, heaviest first.
2. Discover: a strong model reads all lines at once and proposes themes, each with a definition and a kind.
3. Assign: the same model puts every cluster under one theme (or "other"), in batches.
4. Weigh: a neuron's share of the emotion's push is split over its clusters by size; a theme's weight is the sum.

- Every step is cached under results/<run_id>/themes/, so a rerun resumes.
"""

import csv
import json
import re
import time
from pathlib import Path

from pydantic import BaseModel

from .config import ModesRunConfig
from .judge import ask, read_key

DISCOVER = """Below are clusters of what individual neurons in a language model respond to. Together these neurons build the model's internal "{emotion}" direction: the vector that separates {emotion} stories from other stories.

Each line: cluster id | the neuron's logit lens (tokens its output pushes up) | what the cluster's firing tokens have in common | the scenes they sit in | one example with the firing token in [[brackets]].
Lines are ordered by how much of the "{emotion}" direction they carry, heaviest first.

Find the recurring THEMES of meaning: the topics, situations, actions, feelings, writing styles or syntax patterns that many clusters share. These themes show what the model (and the stories it learned from) treats as "{emotion}". Include themes that are not obviously {emotion}, such as recurring story topics, settings or writing habits: those are what we most want to find.

Give {n_min} to {n_max} themes. Answer with a JSON array and nothing else:
[{{"theme": "short name, 2-5 words", "definition": "one sentence: what clusters in it share", "kind": "feeling" or "situation" or "action" or "topic" or "style" or "syntax"}}]"""

ASSIGN = """Themes (name: definition):
{themes}

Put each cluster below under the ONE theme that fits it best, or "other" if none fits. Lines: cluster id | lens | what the firing tokens share | scenes | example.

{lines}

Answer with one JSON object per line and nothing else, one per cluster, in order:
{{"id": "<cluster id>", "theme": "<theme name exactly as given, or other>"}}"""


class Theme(BaseModel):
    theme: str
    definition: str
    kind: str


class ClusterLine(BaseModel):
    id: str
    neuron: int
    weight: float  # % of the emotion's push carried by this cluster
    text: str


def short_example(text: str) -> str:
    i = text.find("[[")
    j = text.find("]]", i) + 2
    left = " ".join(text[max(0, i - 80) : i].split()[-10:])
    right = " ".join(text[j : j + 50].split()[:6])
    return f"{left} {text[i:j]} {right}".strip()


def cluster_lines(cards: dict) -> list[ClusterLine]:
    """Clusters on each set neuron's side, heaviest first."""
    judge = cards["judges"][0]
    lines = []
    for card in cards["neurons"]:
        if card["is_control"]:
            continue
        side = card["side"]
        clusters = card["poles"][side]["clusters"]
        total = sum(c["size"] for c in clusters) or 1
        lens = " ".join(t.strip() for t in card["lens"]["up" if side == "high" else "down"][:6])
        for c in clusters:
            verdict = c["verdicts"].get(judge)
            if verdict is None or not c["examples"]:
                continue
            cid = f"n{card['neuron']}-{side}{c['id']}"
            text = f"{cid} | {lens} | {verdict['mode']} | {verdict['scene']} | {short_example(c['examples'][0][1])}"
            lines.append(ClusterLine(id=cid, neuron=card["neuron"], weight=card["share"] * c["size"] / total, text=text))
    return sorted(lines, key=lambda line: -line.weight)


def json_array(reply: str) -> list:
    match = re.search(r"\[.*\]", reply, re.S)
    assert match, f"no JSON array in the reply: {reply[:200]!r}"
    return json.loads(match.group(0))


def discover(lines: list[ClusterLine], cfg: ModesRunConfig, key: str, path: Path) -> list[Theme]:
    if path.exists():
        return [Theme(**t) for t in json.loads(path.read_text())]
    t = cfg.themes
    prompt = DISCOVER.format(emotion=cfg.concept, n_min=t.n_min, n_max=t.n_max)
    reply = ask(t.model, prompt, "\n".join(line.text for line in lines), key, t.max_tokens)
    themes = [Theme(**x) for x in json_array(reply)]
    path.write_text(json.dumps([x.model_dump() for x in themes], indent=1))
    return themes


def assign(lines: list[ClusterLine], themes: list[Theme], cfg: ModesRunConfig, key: str, folder: Path) -> dict[str, str]:
    """{cluster id: theme}. One cached file per batch; a cluster the model skips stays unassigned."""
    t = cfg.themes
    names = {x.theme for x in themes}
    theme_text = "\n".join(f"- {x.theme}: {x.definition}" for x in themes)
    out = {}
    for b, start in enumerate(range(0, len(lines), t.batch_size)):
        path = folder / f"assign_{b:02d}.jsonl"
        if not path.exists():
            batch = lines[start : start + t.batch_size]
            reply = ask(
                t.model,
                "You label text clusters by theme.",
                ASSIGN.format(themes=theme_text, lines="\n".join(x.text for x in batch)),
                key,
                t.max_tokens,
            )
            rows = [json.loads(m) for m in re.findall(r"\{[^{}]*\}", reply)]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            print(f"[themes] assigned batch {b}: {len(rows)} of {len(batch)} clusters", flush=True)
        for row in map(json.loads, path.read_text().splitlines()):
            if isinstance(row, dict) and "id" in row:
                out[row["id"]] = row.get("theme") if row.get("theme") in names else "other"
    return out


def weigh(lines: list[ClusterLine], themes: list[Theme], assigned: dict[str, str]) -> list[dict]:
    """Per theme: % of the push, clusters, neurons, and its heaviest clusters."""
    rows = {x.theme: {**x.model_dump(), "push_pct": 0.0, "clusters": 0, "neurons": set(), "top": []} for x in themes}
    rows["other"] = {
        "theme": "other",
        "definition": "fits no theme",
        "kind": "-",
        "push_pct": 0.0,
        "clusters": 0,
        "neurons": set(),
        "top": [],
    }
    rows["unassigned"] = {
        "theme": "unassigned",
        "definition": "the model gave no label",
        "kind": "-",
        "push_pct": 0.0,
        "clusters": 0,
        "neurons": set(),
        "top": [],
    }
    for line in lines:
        row = rows[assigned.get(line.id, "unassigned")]
        row["push_pct"] += line.weight
        row["clusters"] += 1
        row["neurons"].add(line.neuron)
        if len(row["top"]) < 6:
            row["top"].append(line.text)
    result = [{**r, "neurons": len(r["neurons"]), "push_pct": round(r["push_pct"], 2)} for r in rows.values()]
    return sorted(result, key=lambda r: -r["push_pct"])


def write_cluster_themes(path: Path, lines: list[ClusterLine], assigned: dict[str, str]) -> None:
    """One row per cluster: id, neuron, pole, cluster, weight (% of the push), theme."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "neuron", "pole", "cluster", "weight_pct", "theme"])
        for line in lines:
            pole, cluster = re.match(r"n\d+-(high|low)(\d+)", line.id).groups()
            writer.writerow(
                [line.id, line.neuron, pole, int(cluster), round(line.weight, 4), assigned.get(line.id, "unassigned")]
            )


def run_themes(cfg: ModesRunConfig) -> list[dict]:
    """The themes step: results/<run_id>/themes.json (needs the report's cards.json)."""
    t0 = time.time()
    folder = cfg.result_dir / "themes"
    folder.mkdir(parents=True, exist_ok=True)
    lines = cluster_lines(json.loads((cfg.result_dir / "cards.json").read_text()))
    key = read_key(cfg.judge.key_file)
    print(f"[themes] {len(lines)} clusters on the {cfg.concept} side of the set's neurons; model {cfg.themes.model}", flush=True)
    themes = discover(lines, cfg, key, folder / "themes.json")
    print(f"[themes] {len(themes)} themes: {', '.join(x.theme for x in themes)}", flush=True)
    assigned = assign(lines, themes, cfg, key, folder)
    result = weigh(lines, themes, assigned)
    (cfg.result_dir / "themes.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    write_cluster_themes(cfg.result_dir / "cluster_themes.csv", lines, assigned)
    print(f"[themes] saved {cfg.result_dir}/themes.json and cluster_themes.csv | {time.time() - t0:.0f}s", flush=True)
    return result
