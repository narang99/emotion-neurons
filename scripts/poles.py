"""The poles of every neuron in the MLPs of chosen layers: quantiles (e.g. p3, p97) over general text, and means. GPU.

    python scripts/poles.py --config configs/poles.yaml

- See `emotion_neurons.poles`. A token of neuron n is at its low pole if `a_n <= p3`, at its high pole if `a_n >= p97`.
- Writes results/<run_id>/thresholds.safetensors: each quantile (e.g. `p3`) and `mean` as [layer, neuron], fp32,
  `layers` [layer] with the layer numbers of the rows; and `meta.json`.
"""

import json
import sys
import time

import torch
from safetensors.torch import save_file

from emotion_neurons.config import PolesRunConfig, load_run_config
from emotion_neurons.models import adapter_for
from emotion_neurons.poles import collect_windows, layer_activations, mean_per_neuron, quantiles_per_neuron

torch.set_grad_enabled(False)


def quantile_name(q: float) -> str:
    return f"p{round(q * 100):g}"


def main():
    cfg = load_run_config(PolesRunConfig, sys.argv[1:])
    t0 = time.time()
    adapter = adapter_for(cfg.model)
    model, tokenizer = adapter.load(cfg.device)
    windows = collect_windows(cfg.poles, tokenizer)
    names = [quantile_name(x) for x in cfg.poles.quantiles]
    per_layer = {name: [] for name in [*names, "mean"]}
    tokens = 0
    for layer in cfg.layers:
        acts = layer_activations(model, adapter, windows, cfg.poles, layer, tokenizer.pad_token_id)
        q = quantiles_per_neuron(acts, cfg.poles.quantiles, model.device)
        for i, name in enumerate(names):
            per_layer[name].append(q[i])
        per_layer["mean"].append(mean_per_neuron(acts, model.device))
        tokens = len(acts)
        print(f"[poles] layer {layer}: {q.shape[1]} neurons over {tokens} tokens, {time.time() - t0:.0f}s")
        del acts
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    tensors = {name: torch.stack(rows).contiguous() for name, rows in per_layer.items()}
    tensors["layers"] = torch.tensor(cfg.layers)
    save_file(tensors, str(cfg.result_dir / "thresholds.safetensors"))
    meta = {"model": cfg.model.name, "layers": cfg.layers, "tokens": tokens, "quantiles": dict(zip(names, cfg.poles.quantiles))}
    (cfg.result_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[saved] {cfg.result_dir}/thresholds.safetensors")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
