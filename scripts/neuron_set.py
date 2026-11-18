"""Stage 3: the layer-L MLP neurons behind the MLP's write for each emotion. CPU, no forward pass.

    python scripts/neuron_set.py --config configs/stage3_l25.yaml

- Method in `emotion_neurons.decomposition.dims_then_neurons`.
- Only per-dimension cut-off counts are stored, since any other threshold rebuilds exactly from W_down and dA.
- Writes results/<run_id>/neuron_set.json:
  - summary per emotion, members per (emotion, dim pole, neuron pole), per kept dimension how many neurons each
    pole needed, and a sweep of set sizes over shares.
"""

import json
import sys

import torch
from jaxtyping import Float
from torch import Tensor

from emotion_neurons.config import load_run_config
from emotion_neurons.decomposition import neuron_terms
from emotion_neurons.decomposition.dims_then_neurons import dims_kept, neurons_for
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.models import load_w_down
from emotion_neurons.vectors import EmotionVectors

POLES = (("pos", 1), ("neg", -1))


def set_size(delta: Tensor, terms: Float[Tensor, "d_model d_mlp"], dim_share: float, neuron_share: float) -> int:
    members = set()
    for _, dim_sign in POLES:
        dims = dims_kept(delta, dim_sign, dim_share)
        if len(dims):
            for _, neuron_sign in POLES:
                members |= set(neurons_for(terms[dims], neuron_sign, neuron_share)[0].tolist())
    return len(members)


def main():
    cfg = load_run_config(sys.argv[1:])
    vectors = EmotionVectors.load(cfg.results_dir / cfg.neuron_set.vectors_run / "vectors.safetensors")
    W = load_w_down(cfg.model, cfg.layer)
    ds, ns = cfg.neuron_set.dim_share, cfg.neuron_set.neuron_share
    print(f"[load] W_down {tuple(W.shape)}; dim_share {ds}, neuron_share {ns}")

    summary, members, counts, sweep = {}, {}, {}, {}
    union = set()
    write, shift = vectors.get("mlp_out"), vectors.get("mlp_act")
    for e, emotion in enumerate(EMOTIONS):
        delta, dA = write[e], shift[e]
        rel = ((W @ dA - delta).norm() / delta.norm()).item()
        assert rel < 0.05, f"{emotion}: W_down @ dA differs from delta by {rel:.3f}"
        terms = neuron_terms(W, dA)
        s = {"rebuild_rel_err": round(rel, 5)}
        emotion_union = set()
        for dim_pole, dim_sign in POLES:
            dims = dims_kept(delta, dim_sign, ds)
            s[f"dims_{dim_pole}"] = len(dims)
            for neuron_pole, neuron_sign in POLES:
                empty = (torch.tensor([], dtype=torch.long), torch.tensor([]))
                ids, k = neurons_for(terms[dims], neuron_sign, ns) if len(dims) else empty
                key = f"{emotion}/{dim_pole}dim/{neuron_pole}"
                members[key] = ids.tolist()
                counts[key] = {"dims": dims.tolist(), "k": k.tolist()}
                s[f"neurons_{dim_pole}dim_{neuron_pole}"] = len(ids)
                s[f"median_k_{dim_pole}dim_{neuron_pole}"] = float(k.float().median()) if len(k) else 0.0
                emotion_union |= set(ids.tolist())
        s["neurons_any"] = len(emotion_union)
        union |= emotion_union
        summary[emotion] = s
        sweep[emotion] = {f"{d}/{n}": set_size(delta, terms, d, n) for d in (0.5, 0.7, 0.9) for n in (0.1, 0.3, 0.5, 0.9)}
        print(f"[{emotion}] err {rel:.4f} | dims +{s['dims_pos']} -{s['dims_neg']} | neurons per dim (median, +dim) "
              f"+{s['median_k_posdim_pos']:.0f} -{s['median_k_posdim_neg']:.0f} | union {s['neurons_any']} | "
              f"sweep {sweep[emotion]}")  # fmt: skip

    print(f"[set] {len(union)} of {W.shape[1]} neurons in the union over emotions and poles")
    out = {"dim_share": ds, "neuron_share": ns, "d_mlp": W.shape[1], "summary": summary, "union_size": len(union),
           "members": members, "counts": counts, "sweep": sweep}  # fmt: skip
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "neuron_set.json").write_text(json.dumps(out))
    print(f"[saved] {cfg.result_dir}/neuron_set.json")


if __name__ == "__main__":
    main()
