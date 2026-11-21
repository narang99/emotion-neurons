"""The poles of every layer-L neuron: p3 and p97 over a sample of general text. GPU.

    python scripts/poles.py --config configs/poles.yaml

- See `emotion_neurons.poles`. A token of neuron n is at its low pole if `a_n <= p3`, at its high pole if `a_n >= p97`.
- Writes results/<run_id>/thresholds.safetensors: `p3`, `p97` [neuron], fp32; and `meta.json`.
"""

import json
import sys
import time

import torch
from safetensors.torch import save_file

from emotion_neurons.config import load_run_config
from emotion_neurons.models import load_model
from emotion_neurons.poles import corpus_activations, quantiles_per_neuron

torch.set_grad_enabled(False)


def quantile_name(q: float) -> str:
    return f"p{round(q * 100):g}"


def main():
    cfg = load_run_config(sys.argv[1:])
    t0 = time.time()
    model, tokenizer = load_model(cfg.model, cfg.device)
    acts = corpus_activations(model, tokenizer, cfg.poles, cfg.layer)
    q = quantiles_per_neuron(acts, cfg.poles.quantiles, model.device)
    names = [quantile_name(x) for x in cfg.poles.quantiles]
    print(f"[poles] {', '.join(names)} of {q.shape[1]} neurons over {len(acts)} tokens, {time.time() - t0:.0f}s")
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    save_file({n: q[i].contiguous() for i, n in enumerate(names)}, str(cfg.result_dir / "thresholds.safetensors"))
    meta = {"model": cfg.model, "layer": cfg.layer, "tokens": len(acts), "quantiles": dict(zip(names, cfg.poles.quantiles))}
    (cfg.result_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[saved] {cfg.result_dir}/thresholds.safetensors")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
