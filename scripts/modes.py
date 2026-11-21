"""Modes of each harvested neuron at both poles, by HDBSCAN on the MLP input at its hits. GPU for the inputs.

    python scripts/modes.py --config configs/harvest_happy_l25.yaml

- Reads results/<run_id>/harvest.json and runs/<run_id>/{corpus.safetensors, sources.json, hits.safetensors}.
- See `emotion_neurons.modes` for the clustering.
- Checks: the activation recomputed from the stored input matches the harvested one, for `modes.checks` hits.
- Writes results/<run_id>/modes.json (per neuron and pole: clusters with size, mean activation, exemplar ids)
  and results/<run_id>/modes/batch_XX.txt (exemplar snippets per cluster, for Haiku to name).
"""

import json
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file

from emotion_neurons.config import load_run_config
from emotion_neurons.harvest import CorpusActs
from emotion_neurons.models import load_model
from emotion_neurons.modes import inputs_at, pole_modes, recomputed_activation

torch.set_grad_enabled(False)
POLES = ("high", "low")


def load_corpus(run_dir) -> CorpusActs:
    t = load_file(str(run_dir / "corpus.safetensors"))
    sources = [tuple(s) for s in json.loads((run_dir / "sources.json").read_text())]
    return CorpusActs(acts=t["acts"], tokens=t["tokens"], window=t["window"], position=t["position"], sources=sources)


def check_inputs(model, layer, corpus, X, hits, column, count, seed) -> float:
    """Largest relative gap between recomputed and harvested activations over `count` random hits."""
    rng = np.random.default_rng(seed)
    keys = list(hits)
    worst = 0.0
    for k in rng.choice(len(keys), min(count, len(keys)), replace=False):
        neuron, _pole = keys[k].split("/")
        t = int(hits[keys[k]][0])
        harvested = float(corpus.acts[t, column[int(neuron)]])
        recomputed = recomputed_activation(model, layer, int(neuron), X[t])
        worst = max(worst, abs(recomputed - harvested) / max(1.0, abs(harvested)))
    return worst


def snippet(tokenizer, corpus, bounds, t, before, after) -> str:
    start, end = bounds[corpus.window[t]].tolist()
    ids = corpus.tokens
    lo, hi = max(start, t - before), min(end, t + after + 1)
    text = tokenizer.decode(ids[lo:t]) + "[[" + tokenizer.decode(ids[t : t + 1]) + "]]" + tokenizer.decode(ids[t + 1 : hi])
    return text.replace("\n", " ")


def batch_files(result, neurons, corpus, column, tokenizer, cfg) -> list[str]:
    """Text for Haiku: per neuron and pole, each cluster's exemplars with their activation."""
    m = cfg.modes
    bounds = corpus.window_bounds()
    files = []
    for s in range(0, len(neurons), m.neurons_per_batch):
        lines = []
        for n in neurons[s : s + m.neurons_per_batch]:
            lines.append(f"##### neuron {n}")
            for pole in POLES:
                r = result[n][pole]
                lines.append(f"=== {pole} pole ({'p97 and above' if pole == 'high' else 'p3 and below'})")
                groups = [(f"cluster {c['id']}", c["exemplars"]) for c in r["clusters"]] + [
                    ("unclustered (sample)", r["other_exemplars"])
                ]
                for name, exemplars in groups:
                    if not exemplars:
                        continue
                    lines.append(f"--- {name}")
                    for t in exemplars:
                        a = float(corpus.acts[t, column[n]])
                        lines.append(f"  [{a:.2f}] {snippet(tokenizer, corpus, bounds, t, m.snippet_before, m.snippet_after)}")
        files.append("\n".join(lines) + "\n")
    return files


def main():
    cfg = load_run_config(sys.argv[1:])
    m = cfg.modes
    t0 = time.time()
    harvest = json.loads((cfg.result_dir / "harvest.json").read_text())
    neurons = harvest["neurons"] + harvest["controls"]
    column = {n: j for j, n in enumerate(neurons)}
    corpus = load_corpus(cfg.run_dir)
    all_hits = load_file(str(cfg.run_dir / "hits.safetensors"))
    hits = {k: v[: m.hits_used] for k, v in all_hits.items()}
    needed = torch.unique(torch.cat(list(hits.values())))
    print(f"[hits] {len(neurons)} neurons x 2 poles, {len(needed)} distinct hit tokens, {time.time() - t0:.0f}s", flush=True)

    model, tokenizer = load_model(cfg.model, cfg.device)
    X = inputs_at(model, corpus, cfg.layer, needed, m.batch_size)
    worst = check_inputs(model, cfg.layer, corpus, X, hits, column, m.checks, seed=0)
    print(f"[check] recomputed vs harvested activation: largest relative gap {worst:.3f} over {m.checks} hits", flush=True)
    assert worst < 0.05, "recomputed activations do not match the harvest"

    result = {}
    for n in neurons:
        result[n] = {}
        for pole in POLES:
            idx = hits[f"{n}/{pole}"]
            a = corpus.acts[idx, column[n]].float().numpy()
            result[n][pole] = pole_modes(idx, X, a, m, seed=n)
    sizes = [len(result[n][p]["clusters"]) for n in neurons for p in POLES]
    print(
        f"[clusters] median {np.median(sizes):.0f} per neuron and pole, {sum(s == 0 for s in sizes)} with none, {time.time() - t0:.0f}s"
    )

    (cfg.result_dir / "modes.json").write_text(json.dumps({str(n): v for n, v in result.items()}))
    out = cfg.result_dir / "modes"
    out.mkdir(parents=True, exist_ok=True)
    files = batch_files(result, neurons, corpus, column, tokenizer, cfg)
    for b, text in enumerate(files):
        (out / f"batch_{b:02d}.txt").write_text(text)
    print(f"[saved] modes.json and {len(files)} batch files, ~{sum(len(f) for f in files) // 4} tokens for Haiku")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
