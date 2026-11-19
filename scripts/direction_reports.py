"""Three directions of each emotion's layer-25 MLP write, and what each one does on real sentences. GPU.

    python scripts/direction_reports.py --config configs/direction_reports.yaml

- Directions, all in the space MLP-25 writes into:
  - `main`: the MLP's emotion write `delta_e = W_down dA_e` (all neurons);
  - `modes`: the part of it from the sparse set's tail firings, `W_down[:, S] (dA_e^high + dA_e^low)[S]`
    (the classes of `tail_check.py`: tokens >= p97 or <= p3 of general text);
  - `remaining`: `main - modes`.
- Per direction: its norm and the cosines between the three; the logit lens (final norm, then `W_U`); and, over the
  held-out stories and the adult texts, the tokens where MLP-25's own output projects most on it (unit direction).
- Example sentences: per token, the projection of MLP-25's output on each direction.
- Writes results/<run_id>/directions.json and results/<run_id>/report.md.
"""

import json
import sys
import time

import torch
from safetensors.torch import load_file

from emotion_neurons.config import load_run_config
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.models import decoder_layers, load_model
from emotion_neurons.models.qwen3 import StopForward, hooks, raise_stop
from emotion_neurons.texts import load_labelled_texts, load_stories

torch.set_grad_enabled(False)
sys.path.insert(0, "scripts")
from tail_check import contrast, corpus_thresholds, story_class_means  # noqa: E402

KINDS = ("main", "modes", "remaining")


def mlp_outputs(model, ids, layer: int) -> torch.Tensor:
    """MLP-25's output at every token [token, d_model]."""
    store = {}
    block = decoder_layers(model)[layer]
    with hooks(
        [
            block.mlp.register_forward_hook(lambda _m, _a, out: store.__setitem__("o", out[0].float())),
            block.register_forward_hook(raise_stop),
        ]
    ):
        try:
            model.model(input_ids=ids, use_cache=False)
        except StopForward:
            pass
    return store["o"]


def lens(model, tokenizer, v: torch.Tensor, k: int) -> list[str]:
    scaled = (model.model.norm.weight.float() * v).to(model.lm_head.weight.dtype)
    return [tokenizer.decode([i]) for i in (model.lm_head.weight @ scaled).float().topk(k).indices.tolist()]


def marked(tokens: list[str], values: list[float], cut: float) -> str:
    """The sentence with each token above `cut` in bold and its value after it."""
    return "".join(f"**{t.strip() or t}**({v:+.1f}) " if v >= cut else t for t, v in zip(tokens, values)).replace("\n", " ")


def main():
    cfg = load_run_config(sys.argv[1:])
    dr = cfg.direction_reports
    t0 = time.time()
    model, tokenizer = load_model(cfg.model, cfg.device)
    lo, hi, _ = corpus_thresholds(model, tokenizer, cfg)
    train = [s for s in load_stories(cfg.data) if not s.heldout and s.emotion in EMOTIONS]
    labels = torch.tensor([EMOTIONS.index(s.emotion) for s in train])
    class_means, _ = story_class_means(model, tokenizer, train, cfg.layer, lo, hi, cfg.data.split_token)
    parts = {c: contrast(class_means[c].double(), labels).float() for c in class_means}
    dA = load_file(str(dr.vectors_file))["mlp_act/mean/raw"].float()
    W = decoder_layers(model)[cfg.layer].mlp.down_proj.weight.float()
    sets = {e: torch.tensor(v["match"]["neurons"]) for e, v in json.loads(dr.sets_file.read_text()).items()}
    print(f"[parts] {time.time() - t0:.0f}s", flush=True)

    directions = {}
    for e, name in enumerate(EMOTIONS):
        S = sets[name]
        main_v = W @ dA[e].to(W.device)
        modes_v = W[:, S.to(W.device)] @ (parts["high"][e] + parts["low"][e])[S].to(W.device)
        directions[name] = {"main": main_v, "modes": modes_v, "remaining": main_v - modes_v}

    texts = [(s.id, s.text) for s in load_stories(cfg.data) if s.heldout and s.emotion in EMOTIONS]
    texts += [(t.id, t.text) for t in load_labelled_texts(dr.texts_dir)]
    units = {e: torch.stack([d[k] / d[k].norm() for k in KINDS]) for e, d in directions.items()}  # [kind, d_model]
    found = {e: {k: [] for k in KINDS} for e in dr.emotions}
    per_text = {}
    for tid, text in texts:
        ids = tokenizer(text, return_tensors="pt")["input_ids"].to(cfg.device)
        out = mlp_outputs(model, ids, cfg.layer)[0]
        toks = [tokenizer.decode([i]) for i in ids[0].tolist()]
        for e in dr.emotions:
            proj = (out @ units[e].T).cpu()  # [token, kind]
            for j, k in enumerate(KINDS):
                for t in range(4, len(toks)):
                    found[e][k].append((float(proj[t, j]), tid, t))
            if tid.startswith((f"{e}-", f"adult-{e}-")):
                per_text.setdefault(e, []).append((tid, toks, proj))
    print(f"[texts] {len(texts)} texts, {time.time() - t0:.0f}s", flush=True)

    text_of = dict(texts)
    out = {}
    lines = ["# Three directions of the layer-25 emotion write", "",
             "- `main`: the MLP's whole emotion write. `modes`: the part from the sparse set's tail firings (>= p97 or <= p3).",
             "  `remaining`: main minus modes.",
             "- Values: projection of MLP-25's output at that token on the unit direction.", ""]  # fmt: skip
    for e in dr.emotions:
        d = directions[e]
        cos = {
            f"{a}-{b}": float(torch.nn.functional.cosine_similarity(d[a], d[b], dim=0))
            for a, b in (("main", "modes"), ("main", "remaining"), ("modes", "remaining"))
        }
        row = {
            "norm": {k: float(d[k].norm()) for k in KINDS},
            "cos": cos,
            "lens": {k: lens(model, tokenizer, d[k], dr.lens_top) for k in KINDS},
            "top_tokens": {},
        }
        lines += [f"## {e}", "", f"- norms: " + ", ".join(f"{k} {v:.2f}" for k, v in row["norm"].items()),
                  "- cosines: " + ", ".join(f"{k} {v:.2f}" for k, v in cos.items()), "", "| direction | logit lens |", "|---|---|"]  # fmt: skip
        lines += [f"| {k} | {', '.join(repr(x) for x in row['lens'][k][:12])} |" for k in KINDS]
        for k in KINDS:
            top = sorted(found[e][k], reverse=True)[: dr.top_examples]
            row["top_tokens"][k] = []
            lines += ["", f"### {e}: tokens where MLP-25 writes most along `{k}`", ""]
            for v, tid, t in top:
                ids = tokenizer(text_of[tid])["input_ids"]
                ctx = (
                    tokenizer.decode(ids[max(0, t - 12) : t])
                    + "[["
                    + tokenizer.decode(ids[t : t + 1])
                    + "]]"
                    + tokenizer.decode(ids[t + 1 : t + 4])
                ).replace("\n", " ")
                row["top_tokens"][k].append({"value": v, "text": tid, "context": ctx})
                lines.append(f"- {v:+.1f} `{tid}`: {ctx}")
        cuts = {k: sorted(v for v, _, _ in found[e][k])[int(dr.mark_quantile * len(found[e][k]))] for k in KINDS}
        row["mark_cuts"] = cuts
        lines += [
            "",
            f"### {e}: example texts (bold: the top {100 * (1 - dr.mark_quantile):.0f}% of all tokens for that direction)",
            "",
        ]
        for tid, toks, proj in per_text.get(e, [])[: dr.examples]:
            lines.append(f"**{tid}**")
            for j, k in enumerate(KINDS):
                lines.append(
                    f"- `{k}` (cut {cuts[k]:+.1f}): "
                    + marked(toks[: dr.example_tokens], proj[: dr.example_tokens, j].tolist(), cuts[k])
                )
            lines.append("")
        out[e] = row
        print(
            f"[{e}] cos {', '.join(f'{k} {v:.2f}' for k, v in cos.items())} | lens modes {row['lens']['modes'][:6]} | remaining {row['lens']['remaining'][:6]}",
            flush=True,
        )
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "directions.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (cfg.result_dir / "report.md").write_text("\n".join(lines))
    print(f"[saved] {cfg.result_dir}/report.md")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
