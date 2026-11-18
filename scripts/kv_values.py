"""MLP neurons as key-value memories (Geva et al. 2021): what each harvested neuron's value promotes, and whether it
predicts the token that follows its key's hits. CPU.

    python scripts/kv_values.py --config configs/kv_values_l25.yaml

Value of neuron n: v_n = W_down[:, n]. Read through the final RMSNorm gain and the unembedding, logits = U (g * v_n)
(RMSNorm only rescales a single vector, so the token ranking is exact for v_n alone; layer 25 of 36, so this is a
logit-lens read). A positive activation adds +v_n, so the top tokens are what the neuron pushes up when it fires.

Agreement (Geva's test): at every p97+ hit of neuron n in the harvest (past skip_first, next token in the same window),
the rank of the actual next token under v_n, as a percentile of the vocabulary (0 = top). Reference: the same next
tokens ranked under the other harvested neurons' values, which controls for how common those tokens are.

results/<run_id>/: values.json (per neuron: top/bottom tokens, agreement), values.md.
"""

import json
import sys
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from emotion_neurons.config import load_run_config
from emotion_neurons.models import load_w_down
from emotion_neurons.values import unembedding, value_logits, vocab_ranks


def main():
    cfg = load_run_config(sys.argv[1:])
    t0 = time.time()
    src_res, src_run = cfg.results_dir / cfg.probe.source_run, cfg.runs_dir / cfg.probe.source_run
    harvest = json.loads((src_res / "harvest.json").read_text())
    neurons, controls = harvest["neurons"], set(harvest["controls"])
    p97 = np.array([harvest["thresholds"][str(n)][0] for n in neurons], dtype=np.float32)

    W = load_w_down(cfg.model, cfg.layer)[:, neurons]
    gain, U = unembedding(cfg.model)
    tok = Tokenizer.from_file(hf_hub_download(cfg.model, "tokenizer.json"))
    logits = value_logits(W, gain, U)
    V = logits.shape[0]
    rank = vocab_ranks(logits)
    pct = rank.float() / V  # 0 = the value's top token
    print(f"[load] {len(neurons)} neurons ({len(controls)} random controls), vocab {V}, {time.time() - t0:.0f}s")

    acts = np.load(src_run / "acts.npy", mmap_mode="r")
    tokens, pos, win = np.load(src_run / "tokens.npy"), np.load(src_run / "pos.npy"), np.load(src_run / "window.npy")
    has_next = np.zeros(len(tokens), bool)
    has_next[:-1] = win[1:] == win[:-1]
    ok = (pos >= cfg.harvest.skip_first) & has_next

    dec = lambda ids: [tok.decode([int(i)]) for i in ids]  # noqa: E731
    out, own_all, cross_all = {}, [], []
    for j, n in enumerate(neurons):
        hit = np.nonzero(ok & (np.asarray(acts[:, j], dtype=np.float32) >= p97[j]))[0]
        nxt = torch.from_numpy(tokens[hit + 1].astype(np.int64))
        P = pct[nxt]  # [hits, N]: next-token percentile under every neuron's value
        own = float(P[:, j].mean())
        others = [k for k in range(len(neurons)) if k != j]
        cross = float(P[:, others].mean())
        top50_own = float((rank[nxt, j] < 50).float().mean())
        top50_cross = float((rank[nxt][:, others] < 50).float().mean())
        top = logits[:, j].topk(15).indices.tolist()
        bot = (-logits[:, j]).topk(10).indices.tolist()
        nt = np.bincount(tokens[hit + 1], minlength=V).argsort()[::-1][:8]
        out[n] = {"control": n in controls, "hits": len(hit), "promotes": dec(top), "suppresses": dec(bot),
                  "next_pct_own": round(own, 4), "next_pct_cross": round(cross, 4),
                  "next_top50_own": round(top50_own, 4), "next_top50_cross": round(top50_cross, 4),
                  "common_next": dec(nt)}  # fmt: skip
        if n not in controls:
            own_all.append(own)
            cross_all.append(cross)
    rd = cfg.result_dir
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "values.json").write_text(json.dumps(out, ensure_ascii=False))
    sel = [n for n in neurons if n not in controls]
    better = sum(out[n]["next_pct_own"] < out[n]["next_pct_cross"] for n in sel)
    print(f"[agreement] selected: next-token percentile under own value {np.mean(own_all):.3f} vs other values "
          f"{np.mean(cross_all):.3f}; own better for {better}/{len(sel)}")  # fmt: skip
    c = [n for n in neurons if n in controls]
    print(f"[agreement] controls: own {np.mean([out[n]['next_pct_own'] for n in c]):.3f} vs others "
          f"{np.mean([out[n]['next_pct_cross'] for n in c]):.3f}; own better for "
          f"{sum(out[n]['next_pct_own'] < out[n]['next_pct_cross'] for n in c)}/{len(c)}")  # fmt: skip
    lines = ["# Values of the harvested neurons (logit lens through the final norm)", "",
             "Per neuron: tokens its value promotes / suppresses; the tokens that most often follow its p97+ hits; "
             "next-token percentile under its own value vs the other neurons' values (0 = top of the vocabulary).", ""]  # fmt: skip
    for n in neurons:
        o = out[n]
        lines.append(f"## {n}{' (random control)' if o['control'] else ''}: {o['hits']} hits | next-token percentile "
                     f"own {o['next_pct_own']:.3f} vs others {o['next_pct_cross']:.3f}")  # fmt: skip
        lines.append(f"- promotes: {' | '.join(repr(t) for t in o['promotes'])}")
        lines.append(f"- suppresses: {' | '.join(repr(t) for t in o['suppresses'])}")
        lines.append(f"- common next tokens at hits: {' | '.join(repr(t) for t in o['common_next'])}")
        lines.append("")
    (rd / "values.md").write_text("\n".join(lines))
    print(f"[saved] {rd}/values.json, values.md")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
