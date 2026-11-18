"""Modes of a neuron: clusters of its p97+ hits by the MLP input that produced them.

- The MLP input (post_attention_layernorm output) at each hit is L2-normalised, reduced by PCA and clustered by HDBSCAN.
- A noise hit joins the nearest cluster centroid by cosine when it is above that cluster's own `assign_quantile`
  of member cosines; the rest stay "other".
- Each cluster keeps the hits closest to its centroid as exemplars, for Haiku to name.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import PreTrainedModel

from .config import ModesConfig
from .harvest import padded_batch
from .models import read_mlp


class HarvestTokens:
    """A harvest run's token arrays, with each window's start offset."""

    def __init__(self, run_dir: Path):
        self.tokens = np.load(run_dir / "tokens.npy")
        self.pos = np.load(run_dir / "pos.npy")
        self.window = np.load(run_dir / "window.npy")
        self.starts = np.zeros(self.window.max() + 2, dtype=np.int64)
        self.starts[1:] = np.cumsum(np.bincount(self.window, minlength=self.window.max() + 1))

    def window_tokens(self, w: int) -> NDArray[np.int32]:
        return self.tokens[self.starts[w] : self.starts[w + 1]]

    def snippet(self, t: int, decode, before: int = 30, after: int = 8) -> tuple[str, str, str]:
        """Short on purpose: Haiku reads these."""
        w = self.window[t]
        lo, hi = max(self.starts[w], t - before), min(self.starts[w + 1], t + after + 1)
        return decode(self.tokens[lo:t].tolist()), decode([int(self.tokens[t])]), decode(self.tokens[t + 1 : hi].tolist())


def inputs_at(
    model: PreTrainedModel, layer: int, harvest: HarvestTokens, token_ids: list[int], batch_size: int, pad_id: int
) -> dict[int, Float[torch.Tensor, " d_model"]]:
    """MLP input at each requested token, from one pass over the windows that hold them, in fp16 on the CPU."""
    by_window = defaultdict(list)
    for t in token_ids:
        by_window[int(harvest.window[t])].append(t)
    windows = sorted(by_window)
    no_neurons = torch.tensor([0], device=model.device)
    out = {}
    for s in tqdm(range(0, len(windows), batch_size), desc="inputs"):
        batch = windows[s : s + batch_size]
        ids, mask = padded_batch([harvest.window_tokens(w).tolist() for w in batch], pad_id)
        x, _ = read_mlp(model, layer, ids.to(model.device), mask.to(model.device), no_neurons)
        for j, w in enumerate(batch):
            for t in by_window[w]:
                out[t] = x[j, t - harvest.starts[w]].cpu().half()
    return out


def neuron_activation(mlp: torch.nn.Module, n: int, x: Float[torch.Tensor, " d_model"]) -> float:
    return (torch.nn.functional.silu(mlp.gate_proj.weight[n].float() @ x) * (mlp.up_proj.weight[n].float() @ x)).item()


def cluster_labels(Z: NDArray[np.float32], m: ModesConfig) -> NDArray[np.int64]:
    if len(Z) < 2 * m.min_cluster_size:
        return np.full(len(Z), -1)
    P = PCA(n_components=min(m.pca_dims, len(Z) - 1, Z.shape[1]), random_state=0).fit_transform(Z)
    return HDBSCAN(min_cluster_size=m.min_cluster_size, min_samples=m.min_samples, copy=True).fit_predict(P)


def unit_centroids(Z: NDArray[np.float32], labels: NDArray[np.int64]) -> dict[int, NDArray[np.float32]]:
    centroids = {}
    for c in sorted(set(labels) - {-1}):
        centroid = Z[labels == c].mean(0)
        centroids[c] = centroid / np.linalg.norm(centroid)
    return centroids


def assign_noise(Z, labels, centroids: dict[int, NDArray[np.float32]], quantile: float) -> NDArray[np.int64]:
    final = labels.copy()
    if not centroids:
        return final
    keys = list(centroids)
    cos = Z @ np.stack(list(centroids.values())).T
    cut = {c: np.quantile(cos[labels == c, i], quantile) for i, c in enumerate(keys)}
    for i in np.nonzero(labels == -1)[0]:
        best = int(cos[i].argmax())
        if cos[i, best] >= cut[keys[best]]:
            final[i] = keys[best]
    return final


def neuron_modes(n: int, hit_tokens: NDArray[np.int64], hit_inputs, hit_acts: NDArray[np.float32], m: ModesConfig) -> dict:
    """Clusters (size, with assigned noise, mean activation, exemplar token ids), and the unclustered share."""
    Z = torch.nn.functional.normalize(torch.stack(hit_inputs).float(), dim=1).numpy()
    labels = cluster_labels(Z, m)
    centroids = unit_centroids(Z, labels)
    final = assign_noise(Z, labels, centroids, m.assign_quantile)
    clusters = []
    for c in sorted(centroids):
        members = np.nonzero(labels == c)[0]
        closest = members[np.argsort(-(Z[members] @ centroids[c]))[: m.exemplars]]
        clusters.append({"id": int(c), "size": int(len(members)), "with_assigned": int((final == c).sum()),
                         "mean_act": float(hit_acts[final == c].mean()), "exemplars": [int(hit_tokens[i]) for i in closest]})  # fmt: skip
    noise = np.nonzero(final == -1)[0]
    rng = np.random.default_rng(n)
    noise_ex = rng.choice(noise, min(m.noise_exemplars, len(noise)), replace=False) if len(noise) else []
    return {"n_hits": int(len(hit_tokens)), "clusters": clusters, "other": int(len(noise)),
            "other_exemplars": [int(hit_tokens[i]) for i in noise_ex]}  # fmt: skip
