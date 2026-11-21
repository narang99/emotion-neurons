"""Modes: clusters of what a neuron reads when it fires at one pole (from tag exp/stage3_l25, now for both poles).

1. The MLP input (post_attention_layernorm output) at every sampled hit token, from one pass over the windows
   that hold them.
2. Per neuron and pole: L2-normalise the inputs, PCA, HDBSCAN.
   - A noise hit joins the nearest cluster when its cosine to that centre is above the cluster's own
     `assign_quantile` of member cosines; the rest stay "other".
3. Per cluster: the exemplars closest to its centre, for Haiku to name.
"""

from collections import defaultdict

import numpy as np
import torch
from jaxtyping import Float, Int
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from torch import Tensor
from transformers import PreTrainedModel

from .config import ModesConfig
from .harvest import CorpusActs
from .models import read_mlp
from .poles import padded_batch


def inputs_at(
    model: PreTrainedModel, corpus: CorpusActs, layer: int, hit_tokens: Int[Tensor, " hit"], batch_size: int
) -> dict[int, Float[Tensor, " d_model"]]:
    """{token index: MLP input}, fp16 on the CPU, re-run from the windows that hold the hits."""
    bounds = corpus.window_bounds()
    by_window = defaultdict(list)
    for t in hit_tokens.tolist():
        by_window[int(corpus.window[t])].append(t)
    windows = sorted(by_window)
    no_neurons = torch.zeros(0, dtype=torch.long, device=model.device)
    out = {}
    for start in range(0, len(windows), batch_size):
        batch = windows[start : start + batch_size]
        seqs = [corpus.tokens[bounds[w, 0] : bounds[w, 1]].tolist() for w in batch]
        ids, mask = padded_batch(seqs, pad_id=0)
        mlp_in, _ = read_mlp(model, layer, ids.to(model.device), mask.to(model.device), no_neurons)
        for j, w in enumerate(batch):
            for t in by_window[w]:
                out[t] = mlp_in[j, t - int(bounds[w, 0])].half().cpu()
        if (start // batch_size) % 200 == 0:
            print(f"[inputs] {start + len(batch)}/{len(windows)} windows", flush=True)
    return out


def recomputed_activation(model: PreTrainedModel, layer: int, neuron: int, x: Float[Tensor, " d_model"]) -> float:
    mlp = model.model.layers[layer].mlp
    x = x.float().to(model.device)
    return float(torch.nn.functional.silu(mlp.gate_proj.weight[neuron].float() @ x) * (mlp.up_proj.weight[neuron].float() @ x))


def cluster_hits(Z: np.ndarray, cfg: ModesConfig) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """HDBSCAN labels, labels after assigning close noise points, and the unit centre of each cluster."""
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    if len(Z) < 2 * cfg.min_cluster_size:
        labels = np.full(len(Z), -1)
    else:
        P = PCA(n_components=min(cfg.pca_dims, len(Z) - 1, Z.shape[1]), random_state=0).fit_transform(Z)
        labels = HDBSCAN(min_cluster_size=cfg.min_cluster_size, min_samples=cfg.min_samples, copy=True).fit_predict(P)
    centres = {}
    for c in sorted(set(labels) - {-1}):
        centre = Z[labels == c].mean(0)
        centres[c] = centre / np.linalg.norm(centre)
    final = labels.copy()
    if centres:
        keys = list(centres)
        cos = Z @ np.stack([centres[c] for c in keys]).T
        cut = {c: np.quantile(cos[labels == c, i], cfg.assign_quantile) for i, c in enumerate(keys)}
        for i in np.nonzero(labels == -1)[0]:
            best = int(cos[i].argmax())
            if cos[i, best] >= cut[keys[best]]:
                final[i] = keys[best]
    return labels, final, {c: v for c, v in centres.items()}


def pole_modes(hits: Int[Tensor, " hit"], X: dict[int, Tensor], a: Float[Tensor, " hit"], cfg: ModesConfig, seed: int) -> dict:
    """Clusters of one neuron's hits at one pole: size, mean activation, exemplar token indices."""
    Z = torch.stack([X[t] for t in hits.tolist()]).float().numpy()
    labels, final, centres = cluster_hits(Z, cfg)
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    clusters = []
    for c, centre in centres.items():
        members = np.nonzero(labels == c)[0]
        closest = members[np.argsort(-(Zn[members] @ centre))[: cfg.exemplars]]
        clusters.append({
            "id": int(c), "size": int(len(members)), "with_assigned": int((final == c).sum()),
            "mean_act": float(a[final == c].mean()), "exemplars": [int(hits[i]) for i in closest],
        })  # fmt: skip
    noise = np.nonzero(final == -1)[0]
    rng = np.random.default_rng(seed)
    other = rng.choice(noise, min(cfg.other_exemplars, len(noise)), replace=False) if len(noise) else []
    return {
        "n_hits": int(len(hits)),
        "clusters": clusters,
        "other": int(len(noise)),
        "other_exemplars": [int(hits[i]) for i in other],
    }
