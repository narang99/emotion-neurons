"""The modes report: one page with a card per harvested neuron. CPU.

    python scripts/modes_report.py --config configs/harvest_happy_l25.yaml

- Reads results/<run_id>/{harvest.json, modes.json, modes/, judgements/}, the greedy set's CSV and tensors, and
  the layer's W_down and the unembedding (cached; downloaded on first use).
- Per neuron: rank, % of the happy push, happy shift and side, cosine of its happy term with the happy write,
  logit lens of its write, and per pole the clusters with their exemplars and each judge's verdict.
- See `emotion_neurons.modes_report` for the neuron classes.
- Writes results/<run_id>/report.html (self-contained) and results/<run_id>/neurons.csv (one row per neuron).
"""

import csv
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from emotion_neurons.config import load_run_config
from emotion_neurons.lens import decode_columns, top_token_ids, unembedding
from emotion_neurons.models import load_w_down
from emotion_neurons.modes_report import neuron_card, read_cluster_texts, read_judgements

torch.set_grad_enabled(False)
TEMPLATE = Path(__file__).resolve().parents[1] / "src/emotion_neurons/templates/modes_report.html"
LENS_TOP = 12


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
    fields = ["neuron", "rank", "share", "shift", "happy_side", "cos", "class", "is_control"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cards)


def main():
    cfg = load_run_config(sys.argv[1:])
    out = cfg.result_dir
    harvest = json.loads((out / "harvest.json").read_text())
    neurons, controls = harvest["neurons"], harvest["controls"]
    modes = {int(k): v for k, v in json.loads((out / "modes.json").read_text()).items()}
    texts = read_cluster_texts(out / "modes")
    judgements = read_judgements(out / "judgements")
    print(f"[input] {len(neurons)} neurons + {len(controls)} controls, judges {list(judgements)}", flush=True)

    T = load_file(str(cfg.harvest.neurons_csv.parents[1] / "tensors.safetensors"))
    d, shift = T["write"][0].double(), T["shift"][0].double()  # happy is row 0
    W = load_w_down(cfg.model, cfg.layer).double()
    info = neuron_info(neurons, controls, set_rows(cfg.harvest.neurons_csv), W, shift, d)
    per_neuron_lens, happy_lens = lenses(cfg.model, W.float(), d.float(), neurons + controls)

    cards = [neuron_card(n, info[n], modes[n], texts, judgements, per_neuron_lens[n]) for n in neurons + controls]
    payload = {"neurons": cards, "happy_write_lens": happy_lens, "judges": list(judgements)}
    # ASCII escapes: partial byte-level tokens decode to U+FFFD, which must not appear raw in the page
    page = TEMPLATE.read_text().replace("__DATA__", json.dumps(payload, ensure_ascii=True).replace("</", "<\\/"))
    (out / "report.html").write_text(page)
    write_neuron_csv(out / "neurons.csv", cards)
    counts = {c: sum(card["class"] == c and not card["is_control"] for card in cards) for c in ("happy", "split", "not happy")}
    print(f"[classes] happy-set neurons by their happy-side clusters: {counts}")
    print(f"[saved] {out}/report.html ({len(page) / 1e6:.1f} MB) and neurons.csv")


if __name__ == "__main__":
    main()
