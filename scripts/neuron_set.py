"""Stage 3: the layer-L MLP neurons behind the MLP's write for each emotion. CPU, no forward pass.

    python scripts/neuron_set.py --config configs/stage3_l25.yaml

The write: delta_e = mlp_out/mean/raw (Stage 2) = W_down @ dA_e, with dA_e = mlp_act/mean/raw.
1. Dimensions of delta_e covering `dim_share` of its positive mass, and separately of its negative mass.
2. For each kept dimension d, the terms W_down[d, n] * dA_e[n]: the neurons covering `neuron_share` of the
   positive terms and, separately, of the negative terms.
3. The set is the union. Every (emotion, dim pole, dim, neuron pole, neuron, share) row is saved.

Writes results/<run_id>/neuron_set.json (summary + set) and neuron_rows.safetensors (all rows).
"""

import json
import sys
from collections import defaultdict

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from emotion_neurons.config import EMOTIONS, load_run_config


def load_w_down(model: str, layer: int) -> torch.Tensor:
    name = f"model.layers.{layer}.mlp.down_proj.weight"
    try:
        index = json.loads(open(hf_hub_download(model, "model.safetensors.index.json")).read())
        shard = index["weight_map"][name]
    except EntryNotFoundError:  # a single-file model has no index
        shard = "model.safetensors"
    with safe_open(hf_hub_download(model, shard), "pt") as f:
        return f.get_tensor(name).float()  # [d_model, d_mlp]


def cover(values: torch.Tensor, share: float) -> torch.Tensor:
    """Indices of the largest `values` (all > 0) whose sum first reaches `share` of the total."""
    order = values.argsort(descending=True)
    csum = values[order].cumsum(0)
    k = int((csum < share * csum[-1]).sum().item()) + 1
    return order[:k]


def main():
    cfg = load_run_config(sys.argv[1:])
    vec = load_file(str(cfg.results_dir / cfg.neuron_set.vectors_run / "vectors.safetensors"))
    W = load_w_down(cfg.model, cfg.layer)
    ds, ns = cfg.neuron_set.dim_share, cfg.neuron_set.neuron_share
    print(f"[load] W_down {tuple(W.shape)}; dim_share {ds}, neuron_share {ns}")

    rows = defaultdict(list)  # emotion, dim_pole, dim, neuron_pole, neuron, share
    summary, members = {}, defaultdict(set)
    for e, emo in enumerate(EMOTIONS):
        delta, dA = vec["mlp_out/mean/raw"][e].double(), vec["mlp_act/mean/raw"][e].double()
        rebuilt = W.double() @ dA
        rel = ((rebuilt - delta).norm() / delta.norm()).item()
        assert rel < 0.05, f"{emo}: W_down @ dA differs from delta by {rel:.3f}"
        T = W.double() * dA[None, :]  # [d_model, d_mlp] terms
        s = {"rebuild_rel_err": round(rel, 5)}
        for dpole, sign in (("pos", 1), ("neg", -1)):
            v = sign * delta
            dims = cover(v.clamp(min=0), ds) if (v > 0).any() else torch.tensor([], dtype=torch.long)
            per_pole = {"pos": set(), "neg": set()}
            for d in dims.tolist():
                for npole, nsign in (("pos", 1), ("neg", -1)):
                    t = (nsign * T[d]).clamp(min=0)
                    if t.sum() == 0:
                        continue
                    idx = cover(t, ns)
                    tot = t.sum()
                    for n in idx.tolist():
                        rows["emotion"].append(e)
                        rows["dim_pole"].append(sign)
                        rows["dim"].append(d)
                        rows["neuron_pole"].append(nsign)
                        rows["neuron"].append(n)
                        rows["share"].append((t[n] / tot).item())
                    per_pole[npole].update(idx.tolist())
            s[f"dims_{dpole}"] = len(dims)
            s[f"neurons_{dpole}dim_pos"] = len(per_pole["pos"])
            s[f"neurons_{dpole}dim_neg"] = len(per_pole["neg"])
            for npole, ids in per_pole.items():
                members[(emo, dpole, npole)] |= ids
        s["neurons_any"] = len(set().union(*[members[(emo, a, b)] for a in ("pos", "neg") for b in ("pos", "neg")]))
        summary[emo] = s
        print(f"[{emo}] rebuild err {rel:.4f} | dims +{s['dims_pos']} -{s['dims_neg']} | neurons "
              f"(+dim: +{s['neurons_posdim_pos']} -{s['neurons_posdim_neg']}) (-dim: +{s['neurons_negdim_pos']} "
              f"-{s['neurons_negdim_neg']}) | any {s['neurons_any']}")  # fmt: skip

    union = set().union(*members.values())
    print(f"[set] {len(union)} of {W.shape[1]} neurons in the union over emotions and poles")
    out = {
        "dim_share": ds,
        "neuron_share": ns,
        "summary": summary,
        "union_size": len(union),
        "members": {f"{e}/{dp}dim/{np_}": sorted(v) for (e, dp, np_), v in members.items()},
    }
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "neuron_set.json").write_text(json.dumps(out))
    save_file(
        {k: torch.tensor(v, dtype=torch.float32 if k == "share" else torch.int32) for k, v in rows.items()},
        str(cfg.result_dir / "neuron_rows.safetensors"),
    )
    print(f"[saved] {cfg.result_dir}/neuron_set.json, neuron_rows.safetensors ({len(rows['neuron'])} rows)")


if __name__ == "__main__":
    main()
