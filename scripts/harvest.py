"""Harvest both poles of the chosen neurons over general text. GPU.

    python scripts/harvest.py --config configs/harvest_happy_l25.yaml

- Neurons: a greedy-set CSV up to `harvest.rebuilt_pct` of its write, plus `harvest.n_controls` random controls.
- See `emotion_neurons.harvest` for the bands, examples and hits.
- Writes:
  - results/<run_id>/harvest.json: neurons, controls, band edges, examples per band and pole (committed);
  - runs/<run_id>/corpus.safetensors: `acts` [token, neuron] fp16, `tokens`, `window`, `position`;
  - runs/<run_id>/sources.json: (corpus, document id) per window;
  - runs/<run_id>/hits.safetensors: `<neuron>/high`, `<neuron>/low` token indices, for the mode clustering.
"""

import gc
import json
import sys
import time

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

from emotion_neurons.config import load_run_config
from emotion_neurons.harvest import (
    CorpusActs,
    band_edges,
    band_names,
    control_neurons,
    load_corpus,
    neuron_examples,
    neurons_until,
    pole_hits,
    read_corpus,
    save_corpus,
)
from emotion_neurons.models import load_model

torch.set_grad_enabled(False)


def corpus_for(cfg, neurons: list[int]) -> CorpusActs:
    """The saved corpus if an earlier run got that far, else one pass to layer L, saved at once.

    The model is freed before returning: sorting activations for the band edges needs the GPU memory.
    """
    path = cfg.run_dir / "corpus.safetensors"
    if path.exists():
        print(f"[corpus] loading the saved pass from {path}", flush=True)
        return load_corpus(cfg.run_dir)
    model, tokenizer = load_model(cfg.model, cfg.device)
    corpus = read_corpus(model, tokenizer, cfg.harvest, cfg.layer, neurons)
    save_corpus(corpus, cfg.run_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return corpus


def main():
    cfg = load_run_config(sys.argv[1:])
    h = cfg.harvest
    t0 = time.time()
    chosen = neurons_until(h.neurons_csv, h.rebuilt_pct)
    n_neurons = AutoConfig.from_pretrained(cfg.model).intermediate_size
    controls = control_neurons(chosen, n_neurons, h.n_controls, h.seed)
    neurons = chosen + controls
    print(f"[neurons] {len(chosen)} from {h.neurons_csv} (to {h.rebuilt_pct}%) + {len(controls)} controls", flush=True)

    corpus = corpus_for(cfg, neurons)
    assert corpus.acts.shape[1] == len(neurons), f"saved corpus has {corpus.acts.shape[1]} neurons, expected {len(neurons)}"
    print(f"[corpus] {len(corpus.tokens)} tokens, {int(corpus.valid.sum())} valid, {time.time() - t0:.0f}s", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    edges = band_edges(corpus, h, torch.device(cfg.device))
    print(f"[edges] both poles, {time.time() - t0:.0f}s", flush=True)
    bounds = corpus.window_bounds()
    generator = torch.Generator().manual_seed(h.seed)
    examples, hits = {}, {}
    for j, n in enumerate(neurons):
        examples[n] = neuron_examples(corpus, bounds, tokenizer, h, j, edges, generator)
        for pole, idx in pole_hits(corpus, j, edges, h.hits_per_pole, generator).items():
            hits[f"{n}/{pole}"] = idx.contiguous()
    print(f"[examples] {len(neurons)} neurons, {time.time() - t0:.0f}s", flush=True)

    save_file(hits, str(cfg.run_dir / "hits.safetensors"))

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    harvest = {
        "model": cfg.model, "layer": cfg.layer, "neurons": chosen, "controls": controls,
        "tokens": len(corpus.tokens), "valid_tokens": int(corpus.valid.sum()), "config": h.model_dump(mode="json"),
        "bands": {"high": band_names(h.high_quantiles), "low": band_names(h.low_quantiles)},
        "edges": {str(n): {pole: e[:, j].tolist() for pole, e in edges.items()} for j, n in enumerate(neurons)},
        "examples": {str(n): v for n, v in examples.items()},
    }  # fmt: skip
    (cfg.result_dir / "harvest.json").write_text(json.dumps(harvest, ensure_ascii=False))
    print(f"[saved] {cfg.result_dir}/harvest.json and {cfg.run_dir}/")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
