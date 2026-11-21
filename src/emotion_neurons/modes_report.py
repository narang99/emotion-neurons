"""Data for the modes report: one card per harvested neuron, with both poles, judge verdicts and the logit lens.

- A neuron's side is the pole the emotion moves it toward: high if its shift for the emotion is positive, else low.
  - The shift is a story-mean difference, so a few strong firings at the other pole can hide behind it
    (neuron 3832 for happy: high-pole joyful firings, but laugh words drive its mean down).
- Neuron class, from the clusters on its side (a cluster is an emotion scene for a judge at 3+ of 5 windows):
  - "match": at least one cluster is an emotion scene for every judge that judged it;
  - "split": no such cluster, but at least one is for one judge;
  - "no match": no cluster on its side is for any judge.
"""

import csv
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from .config import ModesRunConfig
from .emotions import EMOTIONS
from .judge import SCENE_AT, Verdict
from .lens import decode_columns, top_token_ids, unembedding
from .models import load_w_down

TEMPLATE = Path(__file__).parent / "templates" / "modes_report.html"
LENS_TOP = 12


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
    """{model folder name: {(neuron, pole, cluster): verdict as a dict with scene_windows}}."""
    out = {}
    for folder in sorted(p for p in judgements_dir.iterdir() if p.is_dir()):
        verdicts = {}
        for path in sorted(folder.glob("batch_*.jsonl")):
            for line in path.read_text().splitlines():
                v = Verdict.model_validate_json(line)
                verdicts[(v.neuron, v.pole, v.cluster)] = v.model_dump()
        out[folder.name] = verdicts
    return out


def emotion_side(shift: float) -> str:
    return "high" if shift >= 0 else "low"


def neuron_class(clusters: list[dict], judges: list[str]) -> str:
    """A judge with no verdict on a cluster casts no vote there (for happy, Gemini skipped 55 of 4483 every time)."""
    votes = [[c["verdicts"][j]["scene_windows"] >= SCENE_AT for j in judges if j in c["verdicts"]] for c in clusters]
    if any(v and all(v) for v in votes):
        return "match"
    if any(any(v) for v in votes):
        return "split"
    return "no match"


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
    side = emotion_side(info["shift"])
    return {
        "neuron": n, **info, "side": side, "class": neuron_class(poles[side]["clusters"], judges),
        "lens": lens, "poles": poles,
    }  # fmt: skip


def set_rows(csv_path: Path) -> dict[int, dict]:
    return {int(r["neuron"]): r for r in csv.DictReader(csv_path.open())}


def neuron_info(neurons, controls, rows, W, shift, d) -> dict[int, dict]:
    share = shift * (W.T @ d) / (d @ d)
    info = {}
    for n in neurons + controls:
        term = shift[n] * W[:, n]
        info[n] = {
            "rank": int(rows[n]["rank"]) if n in rows else None, "share": round(100 * float(share[n]), 3),
            "shift": round(float(shift[n]), 4), "cos": round(float(torch.cosine_similarity(term, d, 0)), 3),
            "is_control": n in controls,
        }  # fmt: skip
    return info


def lenses(model: str, W, d, neurons: list[int]) -> tuple[dict[int, dict], dict]:
    tokenizer = AutoTokenizer.from_pretrained(model)
    W_U, gamma = unembedding(model, len(tokenizer))
    ids = top_token_ids(W_U, gamma, torch.cat([W[:, neurons], d[:, None]], 1), LENS_TOP)
    up, down = decode_columns(tokenizer, ids["up"]), decode_columns(tokenizer, ids["down"])
    per_neuron = {n: {"up": up[j], "down": down[j]} for j, n in enumerate(neurons)}
    return per_neuron, {"up": up[-1], "down": down[-1]}


def write_neuron_csv(path: Path, cards: list[dict]) -> None:
    fields = ["neuron", "rank", "share", "shift", "side", "cos", "class", "is_control"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cards)


def run_report(cfg: ModesRunConfig) -> dict:
    """The report stage: results/<run_id>/report.html (self-contained) and results/<run_id>/neurons.csv."""
    out = cfg.result_dir
    harvest = json.loads((out / "harvest.json").read_text())
    neurons, controls = harvest["neurons"], harvest["controls"]
    modes = {int(k): v for k, v in json.loads((out / "modes.json").read_text()).items()}
    texts = read_cluster_texts(out / "modes")
    judgements = read_judgements(out / "judgements")
    print(f"[input] {len(neurons)} neurons + {len(controls)} controls, judges {list(judgements)}", flush=True)

    T = load_file(str(cfg.harvest.neurons_csv.parents[1] / "tensors.safetensors"))
    row = EMOTIONS.index(cfg.emotion)  # rows of the greedy-set tensors follow EMOTIONS
    d, shift = T["write"][row].double(), T["shift"][row].double()
    W = load_w_down(cfg.model, cfg.layer).double()
    info = neuron_info(neurons, controls, set_rows(cfg.harvest.neurons_csv), W, shift, d)
    per_neuron_lens, write_lens = lenses(cfg.model, W.float(), d.float(), neurons + controls)

    cards = [neuron_card(n, info[n], modes[n], texts, judgements, per_neuron_lens[n]) for n in neurons + controls]
    payload = {"emotion": cfg.emotion, "neurons": cards, "write_lens": write_lens, "judges": list(judgements)}
    # ASCII escapes: partial byte-level tokens decode to U+FFFD, which must not appear raw in the page
    page = TEMPLATE.read_text().replace("__DATA__", json.dumps(payload, ensure_ascii=True).replace("</", "<\\/"))
    page = page.replace("__TITLE__", f"{cfg.emotion.capitalize()} Neuron Modes").replace("__EMOTION__", cfg.emotion)
    page = page.replace("__PCT__", f"{harvest['config']['rebuilt_pct']:g}")
    (out / "report.html").write_text(page)
    write_neuron_csv(out / "neurons.csv", cards)
    counts = {c: sum(card["class"] == c and not card["is_control"] for card in cards) for c in ("match", "split", "no match")}
    print(f"[classes] {cfg.emotion}-set neurons by the clusters on their side: {counts}")
    print(f"[saved] {out}/report.html ({len(page) / 1e6:.1f} MB) and neurons.csv", flush=True)
    return counts
