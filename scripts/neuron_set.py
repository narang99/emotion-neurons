"""Stage 3: the layer-L MLP neurons behind the MLP's write for each emotion. CPU, no forward pass.

    python scripts/neuron_set.py --config configs/stage3_l25.yaml

The write: delta_e = mlp_out/mean/raw (Stage 2) = W_down @ dA_e, with dA_e = mlp_act/mean/raw.
1. Dimensions of delta_e covering `dim_share` of its positive mass, and separately of its negative mass.
2. For each kept dimension d, the terms W_down[d, n] * dA_e[n]: the neurons covering `neuron_share` of the
   positive terms and, separately, of the negative terms.
3. The set is the union.

Any other threshold can be rebuilt exactly from W_down and dA (both reproducible), so only the per-dimension
cut-off counts are stored, not every (dim, neuron) row.

Writes results/<run_id>/neuron_set.json: summary, members per (emotion, dim pole, neuron pole), and per kept
dimension how many neurons each pole needed; plus a sweep of set sizes over shares.
"""

import json
import sys

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open
from safetensors.torch import load_file

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


def cover_count(sorted_vals: torch.Tensor, share: float) -> torch.Tensor:
    """Rows sorted descending, all >= 0: how many leading entries first reach `share` of the row total."""
    csum = sorted_vals.cumsum(-1)
    return ((csum < share * csum[..., -1:]).sum(-1) + 1).clamp(max=sorted_vals.shape[-1])


def dims_kept(delta: torch.Tensor, sign: int, share: float) -> torch.Tensor:
    v = (sign * delta).clamp(min=0)
    if v.sum() == 0:
        return torch.tensor([], dtype=torch.long)
    vals, order = v.sort(descending=True)
    return order[: int(cover_count(vals, share))]


def neurons_for(T_rows: torch.Tensor, sign: int, share: float):
    """T_rows [n_dims, d_mlp]: per row, the neurons covering `share` of the `sign` terms. Returns (set, counts)."""
    P = (sign * T_rows).clamp(min=0)
    vals, order = P.sort(dim=1, descending=True)
    k = cover_count(vals, share)
    k = torch.where(P.sum(1) > 0, k, torch.zeros_like(k))
    keep = torch.arange(P.shape[1])[None, :] < k[:, None]
    return torch.unique(order[keep]), k


def main():
    cfg = load_run_config(sys.argv[1:])
    vec = load_file(str(cfg.results_dir / cfg.neuron_set.vectors_run / "vectors.safetensors"))
    W = load_w_down(cfg.model, cfg.layer)
    ds, ns = cfg.neuron_set.dim_share, cfg.neuron_set.neuron_share
    print(f"[load] W_down {tuple(W.shape)}; dim_share {ds}, neuron_share {ns}")

    summary, members, counts, sweep = {}, {}, {}, {}
    union = set()
    for e, emo in enumerate(EMOTIONS):
        delta, dA = vec["mlp_out/mean/raw"][e], vec["mlp_act/mean/raw"][e]
        rel = ((W @ dA - delta).norm() / delta.norm()).item()
        assert rel < 0.05, f"{emo}: W_down @ dA differs from delta by {rel:.3f}"
        T = W * dA[None, :]  # [d_model, d_mlp]: neuron n's term in dimension d
        s = {"rebuild_rel_err": round(rel, 5)}
        emo_union = set()
        for dpole, dsign in (("pos", 1), ("neg", -1)):
            dims = dims_kept(delta, dsign, ds)
            s[f"dims_{dpole}"] = len(dims)
            for npole, nsign in (("pos", 1), ("neg", -1)):
                ids, k = neurons_for(T[dims], nsign, ns) if len(dims) else (torch.tensor([], dtype=torch.long), torch.tensor([]))
                key = f"{emo}/{dpole}dim/{npole}"
                members[key] = ids.tolist()
                counts[key] = {"dims": dims.tolist(), "k": k.tolist()}
                s[f"neurons_{dpole}dim_{npole}"] = len(ids)
                s[f"median_k_{dpole}dim_{npole}"] = float(k.float().median()) if len(k) else 0.0
                emo_union |= set(ids.tolist())
        s["neurons_any"] = len(emo_union)
        union |= emo_union
        summary[emo] = s
        # How the set size moves with the two shares (both poles, union over dims).
        sweep[emo] = {}
        for share_d in (0.5, 0.7, 0.9):
            for share_n in (0.1, 0.3, 0.5, 0.9):
                tot = set()
                for dsign in (1, -1):
                    dd = dims_kept(delta, dsign, share_d)
                    if len(dd):
                        for nsign in (1, -1):
                            tot |= set(neurons_for(T[dd], nsign, share_n)[0].tolist())
                sweep[emo][f"{share_d}/{share_n}"] = len(tot)
        print(f"[{emo}] err {rel:.4f} | dims +{s['dims_pos']} -{s['dims_neg']} | neurons per dim (median, +dim) "
              f"+{s['median_k_posdim_pos']:.0f} -{s['median_k_posdim_neg']:.0f} | union {s['neurons_any']} | "
              f"sweep {sweep[emo]}")  # fmt: skip

    print(f"[set] {len(union)} of {W.shape[1]} neurons in the union over emotions and poles")
    out = {"dim_share": ds, "neuron_share": ns, "d_mlp": W.shape[1], "summary": summary, "union_size": len(union),
           "members": members, "counts": counts, "sweep": sweep}  # fmt: skip
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "neuron_set.json").write_text(json.dumps(out))
    print(f"[saved] {cfg.result_dir}/neuron_set.json")


if __name__ == "__main__":
    main()
