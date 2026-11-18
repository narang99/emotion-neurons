"""Dense circuits for the emotion score on held-out stories, split into emotion-set neurons and the rest. GPU.

    python scripts/circuits.py --config configs/circuits_l25.yaml

- Per held-out story of emotion e (method in `emotion_neurons.circuits`):
  - target: the score `u_e . resid_post_L` at every token from `data.split_token` on;
  - every neuron of layers <= L gets its total effect (frozen attention and norms, one backward pass);
  - check: the direct shares sum to the score;
  - circuits of several sizes (top nodes by |total effect|), each checked by mean-ablating every other neuron,
    against random sets of the same size;
  - the nodes of the `report_size` circuit split by which emotions' sparse sets they belong to.
- Writes results/<run_id>/circuits.jsonl (one line per story) and circuits.md (means over stories).
"""

import json
import sys
import time
from collections import defaultdict

import torch
from safetensors.torch import load_file
from tqdm import tqdm

from emotion_neurons.circuits import (
    ablated_score,
    attribute,
    direct_sum,
    emotion_membership,
    faithfulness,
    keep_mask,
    node_scores,
    random_nodes,
    top_nodes,
)
from emotion_neurons.config import RunConfig, load_run_config
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.models import load_model
from emotion_neurons.texts import Story, load_stories


def chosen_stories(cfg: RunConfig) -> list[Story]:
    held = [s for s in load_stories(cfg.data) if s.heldout and s.emotion in EMOTIONS]
    by_emotion = defaultdict(list)
    for s in held:
        by_emotion[s.emotion].append(s)
    return [s for e in EMOTIONS for s in by_emotion[e][: cfg.circuits.stories_per_emotion]]


def membership_split(scores, nodes, membership, own: str) -> dict:
    """Share of the circuit's |effect| on neurons in the story's own emotion set, only other sets, or none."""
    mass = defaultdict(float)
    per_emotion = defaultdict(float)
    for layer, n in nodes:
        value = abs(float(scores[layer, n]))
        emotions = membership.get((layer, n), [])
        kind = "own" if own in emotions else ("other" if emotions else "none")
        mass[kind] += value
        for e in emotions:
            per_emotion[e] += value / len(emotions)
    total = sum(mass.values())
    return {"share": {k: v / total for k, v in mass.items()}, "by_emotion": {e: v / total for e, v in per_emotion.items()}}


def layer_profile(scores, nodes) -> list[float]:
    """Share of the circuit's |effect| in each layer."""
    per_layer = torch.zeros(scores.shape[0])
    for layer, n in nodes:
        per_layer[layer] += abs(float(scores[layer, n]))
    return (per_layer / per_layer.sum()).tolist()


def story_circuit(cfg: RunConfig, model, tokenizer, story: Story, units, means, membership) -> dict:
    c = cfg.circuits
    layer, e = c.target_layer, EMOTIONS.index(story.emotion)
    ids = tokenizer(story.text, return_tensors="pt")["input_ids"].to(cfg.device)
    unit = units[layer, e]
    attribution = attribute(model, ids, layer, unit)
    tokens = torch.arange(ids.shape[1]) >= cfg.data.split_token
    rebuild = float((direct_sum(attribution) - attribution.score).abs().max() / attribution.score.abs().max())
    scores = node_scores(attribution, tokens)
    n_layers, d_mlp = scores.shape
    full = ablated_score(model, ids, layer, unit, means, None, tokens)
    none = ablated_score(model, ids, layer, unit, means, keep_mask([], n_layers, d_mlp), tokens)
    curve = {}
    for k in c.sizes:
        kept = ablated_score(model, ids, layer, unit, means, keep_mask(top_nodes(scores, k), n_layers, d_mlp), tokens)
        rand = ablated_score(
            model, ids, layer, unit, means, keep_mask(random_nodes(n_layers, d_mlp, k, c.seed), n_layers, d_mlp), tokens
        )
        curve[k] = {"circuit": faithfulness(full, none, kept), "random": faithfulness(full, none, rand)}
    nodes = top_nodes(scores, c.report_size)
    return {
        "story": story.id,
        "emotion": story.emotion,
        "score": float(attribution.score[tokens].sum()),
        "score_none": none,
        "direct_rebuild_err": rebuild,
        "effect_neurons": float(scores.sum()),
        "faithfulness": curve,
        "membership": membership_split(scores, nodes, membership, story.emotion),
        "layers": layer_profile(scores, nodes),
        "top": [[layer_, n, round(float(scores[layer_, n]), 4), membership.get((layer_, n), [])] for layer_, n in nodes[:50]],
    }


def summary(rows: list[dict], cfg: RunConfig) -> str:
    def mean(values):
        return sum(values) / len(values)

    lines = [
        f"# Circuits for the layer-{cfg.circuits.target_layer} emotion score ({len(rows)} held-out stories)",
        "",
        f"- direct shares rebuild the score to max rel err {max(r['direct_rebuild_err'] for r in rows):.3f}.",
        "",
        "| circuit size | faithfulness | random set of that size |",
        "|---|---|---|",
    ]
    for k in cfg.circuits.sizes:
        lines.append(
            f"| {k} | {mean([r['faithfulness'][k]['circuit'] for r in rows]):.2f} | {mean([r['faithfulness'][k]['random'] for r in rows]):.2f} |"
        )
    lines += ["", f"Top {cfg.circuits.report_size} nodes by |effect|, share of their |effect|:", ""]
    for kind in ("own", "other", "none"):
        lines.append(f"- in the story's own emotion set: {mean([r['membership']['share'].get(kind, 0) for r in rows]):.2f}"
                     if kind == "own" else f"- {kind}: {mean([r['membership']['share'].get(kind, 0) for r in rows]):.2f}")  # fmt: skip
    lines += ["", "Share by emotion set, per story emotion (rows: story emotion; columns: set):", ""]
    lines.append("| story | " + " | ".join(e[:5] for e in EMOTIONS) + " |")
    lines.append("|---|" + "---|" * len(EMOTIONS))
    for e in EMOTIONS:
        group = [r for r in rows if r["emotion"] == e]
        if group:
            lines.append(
                f"| {e} | "
                + " | ".join(f"{mean([r['membership']['by_emotion'].get(x, 0) for r in group]):.2f}" for x in EMOTIONS)
                + " |"
            )
    return "\n".join(lines) + "\n"


def main():
    cfg = load_run_config(sys.argv[1:])
    t0 = time.time()
    stored = load_file(str(cfg.circuits.vectors_file))
    units, means = stored["unit"], stored["mlp_act_mean"]
    membership = emotion_membership(cfg.circuits.sparse_layers_file, cfg.circuits.target_layer + 1)
    model, tokenizer = load_model(cfg.model, cfg.device)
    stories = chosen_stories(cfg)
    print(f"[load] {len(stories)} stories, target layer {cfg.circuits.target_layer}, {time.time() - t0:.0f}s")
    out = cfg.result_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out / "circuits.jsonl", "w") as f:
        for story in tqdm(stories, desc="circuits"):
            row = story_circuit(cfg, model, tokenizer, story, units, means, membership)
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            fk = {k: round(v["circuit"], 2) for k, v in row["faithfulness"].items()}
            print(f"[{story.id}] rebuild err {row['direct_rebuild_err']:.3f} | faithfulness {fk} | own-set share "
                  f"{row['membership']['share'].get('own', 0):.2f}", flush=True)  # fmt: skip
    (out / "circuits.md").write_text(summary(rows, cfg))
    print(f"[saved] {out}/circuits.jsonl, circuits.md")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
