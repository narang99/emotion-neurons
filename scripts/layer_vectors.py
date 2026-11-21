"""Each layer's emotion unit vectors and 12-emotion mean activations, as one small file. CPU.

    python scripts/layer_vectors.py --config configs/layers_all.yaml

- Reads the all-layer pooled sums (runs/<run_id>, 5 GB) once; GPU runs then need only this file (about 10 MB).
- results/<run_id>/layer_vectors.safetensors:
  - `unit` [layer, emotion, d_model]: the denoised `resid_post` vector of each layer, unit length;
  - `mlp_act_mean` [layer, d_mlp]: each neuron's mean activation over the 12 emotions (train stories), the
    value a neuron is held at when it is ablated.
"""

import sys

import torch
from safetensors.torch import save_file

from emotion_neurons.capture import LayerMeans
from emotion_neurons.config import load_run_config
from emotion_neurons.journey import layer_vectors
from emotion_neurons.texts import StorySplit, load_story_records
from emotion_neurons.vectors import emotion_means, unit


def main():
    cfg = load_run_config(sys.argv[1:])
    means = LayerMeans(cfg.run_dir)
    split = StorySplit.of(load_story_records(cfg.run_dir))
    units, act_means = [], []
    for layer in range(means.n_layers):
        units.append(unit(layer_vectors(means, layer, split, cfg.vectors).get("resid_post", "mean", "denoised")))
        act_means.append(emotion_means(means.site(layer, "mlp_act"), split, split.train).mean(0))
    out = cfg.result_dir / "layer_vectors.safetensors"
    save_file({"unit": torch.stack(units).float(), "mlp_act_mean": torch.stack(act_means).float()}, str(out))
    print(f"[saved] {out}: unit {tuple(torch.stack(units).shape)}, mlp_act_mean {tuple(torch.stack(act_means).shape)}")


if __name__ == "__main__":
    main()
