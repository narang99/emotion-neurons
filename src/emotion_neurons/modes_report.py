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

from .config import ModesRunConfig
from .emotions import EMOTIONS
from .judge import SCENE_AT, read_verdict
from .lens import decode_columns, top_token_ids, unembedding
from .models import load_tokenizer, load_w_down

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
                v = read_verdict(line)
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


def rounded(x: float | None) -> float | None:
    return None if x is None else round(x, 1)


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
                "bands": c.get("bands"), "pole_pct": rounded(c.get("pole_pct")), "push_pct": rounded(c.get("push_pct")),
                "examples": texts.get(key, []),
                "verdicts": {j: judgements[j][key] for j in judges if key in judgements[j]},
            })  # fmt: skip
        poles[pole] = {"n_hits": modes[pole]["n_hits"], "other": modes[pole]["other"], "clusters": clusters,
                       "band_tokens": modes[pole].get("band_tokens"), "band_hits": modes[pole].get("band_hits"),
                       "other_bands": modes[pole].get("other_bands"), "other_pct": rounded(modes[pole].get("other_pct")),
                       "other_examples": texts.get((n, pole, "other"), [])}  # fmt: skip
    side = emotion_side(info["shift"])
    return {
        "neuron": n, **info, "side": side, "class": neuron_class(poles[side]["clusters"], judges),
        "lens": lens, "poles": poles,
    }  # fmt: skip


def vector_row(greedy_dir: Path, concept: str) -> int:
    """Row of the concept in the greedy set's tensors: concepts.json lists the rows; without it, the EMOTIONS order."""
    listing = greedy_dir / "concepts.json"
    return (json.loads(listing.read_text()) if listing.exists() else list(EMOTIONS)).index(concept)


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
    tokenizer = load_tokenizer(model)
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


def corpus_phrase(harvest: dict) -> str:
    names = ", ".join(c["name"] for c in harvest["config"]["corpora"])
    return f"{harvest['tokens'] / 1e6:.1f}M tokens ({names})"


def sampling_phrase(h: dict) -> str:
    if h.get("hits_per_band") is None:
        return "are sampled uniformly over the pole, so about two thirds come from p97-p99 (p3-p1) and 0.3% from p99.99+ (p0.01-)"
    return (
        f"are sampled up to {h['hits_per_band']} per band (p97-p99, p99-p99.9, p99.9-p99.99, p99.99+; the same at the low pole), "
        "so the rare strongest firings get clusters too; each hit is weighted back by its band's size, so a cluster's "
        "share of the pole and of its push describe the whole pole. The strip under each cluster shows its hits by band, "
        "palest p97-p99 (p3-p1), darkest p99.99+ (p0.01-)"
    )


def scene_classes(cards: dict, judgements: dict[str, dict[tuple, dict]]) -> list[dict]:
    """Each harvested neuron's class under another scene's verdicts, from the clusters on its side.

    - The side and the clusters come from the report's cards, so both scenes judge the same clusters.
    """
    judges = list(judgements)
    rows = []
    for card in cards["neurons"]:
        n, side = card["neuron"], card["side"]
        clusters = [
            {"verdicts": {j: judgements[j][(n, side, c["id"])] for j in judges if (n, side, c["id"]) in judgements[j]}}
            for c in card["poles"][side]["clusters"]
        ]
        rows.append({
            "neuron": n, "rank": card["rank"], "share": card["share"], "is_control": card["is_control"],
            "own_class": card["class"], "class": neuron_class(clusters, judges),
        })  # fmt: skip
    return rows


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
    row = vector_row(cfg.harvest.neurons_csv.parents[1], cfg.concept)
    d, shift = T["write"][row].double(), T["shift"][row].double()
    W = load_w_down(cfg.model, cfg.layer).double()
    info = neuron_info(neurons, controls, set_rows(cfg.harvest.neurons_csv), W, shift, d)
    per_neuron_lens, write_lens = lenses(cfg.model, W.float(), d.float(), neurons + controls)

    cards = [neuron_card(n, info[n], modes[n], texts, judgements, per_neuron_lens[n]) for n in neurons + controls]
    payload = {
        "emotion": cfg.concept, "neurons": cards, "write_lens": write_lens, "judges": list(judgements), "bands": harvest["bands"],
    }  # fmt: skip
    # ASCII escapes: partial byte-level tokens decode to U+FFFD, which must not appear raw in the page
    page = TEMPLATE.read_text().replace("__DATA__", json.dumps(payload, ensure_ascii=True).replace("</", "<\\/"))
    page = page.replace("__TITLE__", f"{cfg.concept.capitalize()} Neuron Modes").replace("__EMOTION__", cfg.concept)
    page = page.replace("__PCT__", f"{harvest['config']['rebuilt_pct']:g}").replace("__LAYER__", str(cfg.layer))
    page = page.replace("__CORPUS__", corpus_phrase(harvest)).replace("__SAMPLING__", sampling_phrase(harvest["config"]))
    (out / "report.html").write_text(page)
    (out / "cards.json").write_text(json.dumps(payload))  # the same data, for scripts (e.g. emotion_neurons.themes)
    write_neuron_csv(out / "neurons.csv", cards)
    counts = {c: sum(card["class"] == c and not card["is_control"] for card in cards) for c in ("match", "split", "no match")}
    print(f"[classes] {cfg.concept}-set neurons by the clusters on their side: {counts}")
    print(f"[saved] {out}/report.html ({len(page) / 1e6:.1f} MB) and neurons.csv", flush=True)
    return counts
