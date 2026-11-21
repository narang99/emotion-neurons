"""What the neurons behind a direction do at the position the direction is read from (e.g. refusal: the last
post-instruction token), from their activations on every train and val instruction.

- Per top neuron (greedy-set rank): its share of the MLP's write, its mean activation on target vs other prompts,
  how well it alone separates held-out target from other prompts (chance a val target prompt is further along its
  side than a val other prompt), its logit lens, its harvest clusters, and the prompts at its extremes.
- One label per neuron from a strong model (OpenRouter), naming what its most pushed prompts share. Cached.
- Writes results/<direction run>/prompt_view.json and prompt_view.html.
"""

import csv
import html
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

from .config import DirectionRunConfig
from .datasets import InstructionSet
from .judge import ask, read_key

LABEL_PROMPT = """Below are neurons from a language model. For each neuron you see the instructions that push it furthest (TOP) and least (BOTTOM), measured at the moment the assistant is about to reply. Some instructions are harmful requests, some harmless.

For each neuron, say in at most 12 words what the TOP instructions share that the BOTTOM ones lack (a topic, a kind of request, a wording, harmfulness, ...). Be concrete.

Answer with one JSON object per line and nothing else: {"neuron": N, "label": "..."}"""


def chance_above(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = a[:, None] - b[None, :]
    return float((diff > 0).double().mean() + 0.5 * (diff == 0).double().mean())


def neuron_view(n: int, row: dict, acts: torch.Tensor, examples, target: str, k: int) -> dict:
    """`acts` is this neuron's activation per example, signed so that the shift (target - other) is positive."""
    is_target = torch.tensor([e.category == target for e in examples])
    val = torch.tensor([e.split == "val" for e in examples])
    train = ~val
    order = acts.argsort(descending=True).tolist()
    return {
        "neuron": n, "rank": int(row["rank"]), "share": float(row["write_share_pct"]), "shift": float(row["shift"]),
        "mean_target": float(acts[is_target & train].mean()), "mean_other": float(acts[~is_target & train].mean()),
        "separation": chance_above(acts[is_target & val], acts[~is_target & val]),
        "top": [{"text": examples[i].text, "category": examples[i].category, "act": float(acts[i])} for i in order[:k]],
        "bottom": [{"text": examples[i].text, "category": examples[i].category, "act": float(acts[i])} for i in order[-k:][::-1]],
    }  # fmt: skip


def neuron_block(v: dict) -> str:
    top = "\n".join(f"  TOP: {p['text'][:160]}" for p in v["top"][:8])
    bottom = "\n".join(f"  BOTTOM: {p['text'][:160]}" for p in v["bottom"][:8])
    return f"neuron {v['neuron']}\n{top}\n{bottom}"


def parse_labels(reply: str) -> dict[int, str]:
    labels = {}
    for m in re.findall(r"\{[^{}]*\}", reply):
        try:
            row = json.loads(m)
            labels[int(row["neuron"])] = str(row["label"])
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return labels


def labels_for(views: list[dict], cache: Path, model: str, key: str) -> dict[int, str]:
    """One label per neuron; neurons the model skips are asked again once. Cached."""
    labels = {int(k): v for k, v in json.loads(cache.read_text()).items()} if cache.exists() else {}
    for _ in range(2):
        missing = [v for v in views if not labels.get(v["neuron"])]
        if not missing:
            break
        labels.update(parse_labels(ask(model, LABEL_PROMPT, "\n\n".join(neuron_block(v) for v in missing), key, 8000)))
    cache.write_text(json.dumps(labels, indent=1))
    return labels


def build_prompt_view(
    cfg: DirectionRunConfig, harvest_run: str, top_n: int = 40, k: int = 6, label_model: str = "anthropic/claude-sonnet-5.5"
) -> dict:
    out = cfg.result_dir
    concept = cfg.neurons.concept
    rows = list(csv.DictReader((out / "neurons" / f"{concept}.csv").open()))[:top_n]
    examples = InstructionSet(cfg.instructions).examples()
    acts_all = load_file(str(out / "neuron_acts.safetensors"))["acts"].float()
    assert acts_all.shape[0] == len(examples), "neuron_acts rows must match the example order"
    cards = {c["neuron"]: c for c in json.loads((cfg.results_dir / harvest_run / "cards.json").read_text())["neurons"]}
    views = []
    for row in rows:
        n = int(row["neuron"])
        sign = 1.0 if float(row["shift"]) >= 0 else -1.0
        v = neuron_view(n, row, sign * acts_all[:, n], examples, cfg.direction.target, k)
        card = cards.get(n)
        if card:
            side = card["side"]
            v["lens"] = card["lens"]["up" if side == "high" else "down"][:10]
            v["clusters"] = [
                next(iter(c["verdicts"].values()))["mode"] for c in card["poles"][side]["clusters"] if c["verdicts"]
            ][:4]
        views.append(v)
    labels = labels_for(views, out / "prompt_view_labels.json", label_model, read_key(Path("~/.config/openrouter/key")))
    for v in views:
        v["label"] = labels.get(v["neuron"], "")
    summary = {
        "concept": concept, "target": cfg.direction.target, "top_n": top_n,
        "separating_neurons": sum(v["separation"] >= 0.8 for v in views),
        "share_of_top": round(sum(v["share"] for v in views), 2), "neurons": views,
    }  # fmt: skip
    (out / "prompt_view.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    page = render(summary, json.loads((out / "direction.json").read_text()), json.loads((out / "neurons.json").read_text()))
    (out / "prompt_view.html").write_text(page.replace("\ufffd", "&#xFFFD;"))  # partial byte-level tokens decode to U+FFFD
    return summary


STYLE = """
:root { --bg: #f6f7f5; --panel: #fff; --fg: #1d2421; --muted: #5d6a64; --line: #dde2de; --target: #a33a2a; --other: #2b6cb0; --accent: #1f6f5c;
  --sans: "IBM Plex Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #121614; --panel: #1a201d; --fg: #e4e9e6; --muted: #9aa8a1;
  --line: #2c3530; --target: #f08a78; --other: #7fb2ea; --accent: #5fc3a6; color-scheme: dark } }
:root[data-theme="dark"] { --bg: #121614; --panel: #1a201d; --fg: #e4e9e6; --muted: #9aa8a1; --line: #2c3530; --target: #f08a78;
  --other: #7fb2ea; --accent: #5fc3a6; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--sans); font-size: 14px; line-height: 1.5; }
.wrap { max-width: 1000px; margin: 0 auto; padding-block: 24px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 16px; }
h1 { font-size: 22px; margin: 0; text-wrap: balance; } h2 { font-size: 16px; margin: 0; }
p { margin: 0; color: var(--muted); max-width: 80ch; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }
td, th { padding: 3px 14px 3px 0; border-bottom: 1px solid var(--line); text-align: left; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; display: grid; gap: 8px; min-width: 0; }
.head { display: flex; flex-wrap: wrap; gap: 4px 14px; align-items: baseline; }
.nid { font-family: var(--mono); font-weight: 500; } .meta { color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums; }
.label { font-weight: 600; } .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; min-width: 0; }
.col h3 { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 0 0 4px; }
ul { margin: 0; padding-left: 16px; display: grid; gap: 3px; } li { overflow-wrap: anywhere; }
.harmful { color: var(--target); } .harmless { color: var(--other); }
.toks { font-family: var(--mono); font-size: 12px; color: var(--muted); }
@media (max-width: 700px) { .cols { grid-template-columns: 1fr; } }
"""


def render(summary: dict, direction: dict, neurons: dict) -> str:
    rates = direction["refusal_rates"]
    target = summary["target"]

    def prompt_list(items: list[dict]) -> str:
        return "".join(
            f'<li><span class="{html.escape(p["category"])}">[{html.escape(p["category"])}]</span> {html.escape(p["text"][:200])}</li>'
            for p in items
        )

    cards = []
    for v in summary["neurons"]:
        lens = " ".join(html.escape(t.strip()) for t in v.get("lens", []))
        clusters = "; ".join(html.escape(c) for c in v.get("clusters", []))
        cards.append(f"""<div class="card"><div class="head"><span class="nid">#{v["rank"]} n{v["neuron"]}</span>
<span class="label">{html.escape(v["label"])}</span></div>
<div class="meta">{v["share"]:.2f}% of the write · mean activation {target} {v["mean_target"]:+.2f} vs other {v["mean_other"]:+.2f}
(signed along its shift) · alone separates held-out prompts: {100 * v["separation"]:.0f}%</div>
{f'<div class="toks">lens: {lens}</div>' if lens else ""}{f'<div class="meta">general-text clusters (harvest): {clusters}</div>' if clusters else ""}
<div class="cols"><div class="col"><h3>pushed most</h3><ul>{prompt_list(v["top"])}</ul></div>
<div class="col"><h3>pushed least</h3><ul>{prompt_list(v["bottom"])}</ul></div></div></div>""")
    rate_rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{100 * v:.1f}%</td></tr>" for k, v in rates.items())
    choice = direction["choice"]
    return f"""<title>Refusal Neurons at the Reply Token</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{STYLE}</style>
<div class="wrap">
<h1>What the neurons behind {html.escape(direction["model"])}'s refusal direction do at the reply token</h1>
<p>Direction: harmful minus harmless instructions (the paper's splits, {direction["n_train"]} train + {direction["n_val"]} val per side),
residual stream at layer {choice["layer"]}, post-instruction position {choice["position"]}, chosen as in Arditi et al. 2024. Neurons: the
MLP of layer {neurons["layer"]}, the last one writing into it; {neurons["set_size"]} of {neurons["d_mlp"]} neurons rebuild 90% of its
refusal write (cosine of that write with the direction {neurons["cos_mlp_write_with_direction"]:.2f}).</p>
<h2>The direction works (greedy replies, val prompts)</h2>
<table><tr><th>condition</th><th>refusal rate</th></tr>{rate_rows}</table>
<h2>The top {summary["top_n"]} neurons ({summary["share_of_top"]:.1f}% of the write), read at the reply token</h2>
<p>Activations are signed so that "pushed most" means furthest toward refusal. {summary["separating_neurons"]} of these
{summary["top_n"]} neurons alone separate held-out harmful from harmless prompts 80% of the time or more. Labels: Sonnet 5.5,
from the prompts at each neuron's extremes.</p>
{"".join(cards)}
</div>"""
