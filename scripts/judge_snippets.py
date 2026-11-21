"""Context snippets for blind judging of the selected neurons. GPU.

    python scripts/judge_snippets.py --config configs/judge_sets.yaml

- Reads results/<run_id>/selection.json (from `judge_select.py`).
- Per neuron, its strongest firings on the general corpus (top and bottom), and its strongest firings on the train
  stories of all 12 emotions and neutral. Story labels are not shown.
- Writes one file per neuron, named by its blind id only, to results/<run_id>/snippets/. No neuron number, no group.
"""

import json
import sys

import torch

from emotion_neurons.config import load_run_config
from emotion_neurons.harvest import collect_windows, padded_batch
from emotion_neurons.models import load_model, read_mlp
from emotion_neurons.texts import load_stories

torch.set_grad_enabled(False)


def context(tokenizer, toks: list[int], p: int) -> str:
    text = (
        tokenizer.decode(toks[max(0, p - 14) : p])
        + " [["
        + tokenizer.decode(toks[p : p + 1])
        + "]] "
        + tokenizer.decode(toks[p + 1 : p + 4])
    )
    return " ".join(text.split())


def corpus_acts(model, tokenizer, cfg, idx) -> tuple[torch.Tensor, list[tuple[int, int]], list[list[int]]]:
    h = cfg.harvest
    windows = collect_windows(h, tokenizer)
    values, where = [], []
    for start in range(0, len(windows.tokens), h.batch_size):
        ids, mask = padded_batch(windows.tokens[start : start + h.batch_size], tokenizer.pad_token_id)
        _, a = read_mlp(model, cfg.layer, ids.to(model.device), mask.to(model.device), idx)
        keep = mask.bool()
        keep[:, : h.skip_first] = False
        values.append(a[keep.to(a.device)].float().cpu())
        where += [(start + int(w), int(p)) for w, p in keep.nonzero().tolist()]
    return torch.cat(values), where, windows.tokens


def story_acts(model, tokenizer, cfg, idx) -> tuple[torch.Tensor, list[tuple[int, int]], list[list[int]]]:
    stories = [s for s in load_stories(cfg.data) if not s.heldout]
    values, where, toks = [], [], []
    for i, s in enumerate(stories):
        ids = tokenizer(s.text, return_tensors="pt")["input_ids"]
        _, a = read_mlp(model, cfg.layer, ids.to(model.device), torch.ones_like(ids).to(model.device), idx)
        values.append(a[0, 4:].float().cpu())
        where += [(i, p) for p in range(4, ids.shape[1])]
        toks.append(ids[0].tolist())
    return torch.cat(values), where, toks


def main():
    cfg = load_run_config(sys.argv[1:])
    js = cfg.judge_sets
    picks = json.loads((cfg.result_dir / "selection.json").read_text())
    neurons = sorted({p["neuron"] for p in picks})
    col = {n: j for j, n in enumerate(neurons)}
    model, tokenizer = load_model(cfg.model, cfg.device)
    idx = torch.tensor(neurons, device=model.device)
    c_vals, c_where, c_toks = corpus_acts(model, tokenizer, cfg, idx)
    print(f"[corpus] {len(c_vals)} tokens", flush=True)
    s_vals, s_where, s_toks = story_acts(model, tokenizer, cfg, idx)
    print(f"[stories] {len(s_vals)} tokens", flush=True)

    out_dir = cfg.result_dir / "snippets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in picks:
        j = col[p["neuron"]]
        lines = [f"# {p['id']}", "", f"Emotion to judge against: **{p['emotion']}**", "",
                 "## Strongest positive firings on general text", ""]  # fmt: skip
        lines += [
            f"{r + 1}. {context(tokenizer, c_toks[c_where[i][0]], c_where[i][1])}"
            for r, i in enumerate(c_vals[:, j].topk(js.corpus_top).indices.tolist())
        ]
        lines += ["", "## Strongest negative firings on general text", ""]
        lines += [
            f"{r + 1}. {context(tokenizer, c_toks[c_where[i][0]], c_where[i][1])}"
            for r, i in enumerate((-c_vals[:, j]).topk(js.corpus_bottom).indices.tolist())
        ]
        lines += ["", "## Strongest positive firings in short stories (mixed topics and moods)", ""]
        lines += [
            f"{r + 1}. {context(tokenizer, s_toks[s_where[i][0]], s_where[i][1])}"
            for r, i in enumerate(s_vals[:, j].topk(js.story_top).indices.tolist())
        ]
        (out_dir / f"{p['id']}.md").write_text("\n".join(lines) + "\n")
    print(f"[saved] {len(picks)} snippet files in {out_dir}")


if __name__ == "__main__":
    main()
