"""Sparse neuron sets per emotion write: hill-climb the size / direction trade-off. CPU, no forward pass.

    python scripts/sparse_set.py --config configs/sparse_l25.yaml

- Setting and moves: `emotion_neurons.decomposition.sparse`; orderings: `...decomposition.selection`.
- Target `raw` is the write; `denoised` removes the write's neutral PCs from the write and every u_n (W -> P W).
- Per emotion and ordering method: the order, its train cos and held-out transfer curves, the smallest k whose
  transfer matches the full set (`ceiling`), and the best transfer.
- Per target cos tau: the smallest greedy prefix, and the set after shrinking with swaps.
- Writes results/<run_id>/sparse_set.json.
"""

import json
import sys
import time

import torch
from jaxtyping import Float
from torch import Tensor

from emotion_neurons.config import SparseSetConfig, load_run_config
from emotion_neurons.decomposition import ActivationShifts, activation_shifts, greedy, prefix_curves, set_metrics, shrink
from emotion_neurons.decomposition import selection as sel
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.models import load_w_down
from emotion_neurons.vectors import EmotionVectors

torch.set_grad_enabled(False)


def target_weights(W: Float[Tensor, "d_model d_mlp"], vectors: EmotionVectors, target: str) -> Float[Tensor, "d_model d_mlp"]:
    if target == "raw":
        return W
    pcs = vectors.neutral_pcs("mlp_out").float()
    return W - pcs.T @ (pcs @ W)


def orders_for(W, WtW, shifts: ActivationShifts, e: int, sp: SparseSetConfig) -> tuple[dict[str, list[int]], Tensor, Tensor]:
    a = shifts.train[e]
    d = W @ a
    G = sel.gram(WtW, a)
    c = sel.plain(W, a)
    orders = {"greedy": greedy(G, c, sp.greedy_steps)[0]}
    G_half, c_cross = sel.crossfit(WtW, shifts.half_a[e], shifts.half_b[e], d)
    orders["crossfit"] = greedy(G_half, c_cross, sp.greedy_steps)[0]
    del G_half
    for kappa in sp.shrink_kappas:
        orders[f"shrunk{kappa:g}"] = greedy(G, sel.shrunk(W, a, shifts.std_error[e], kappa), sp.greedy_steps)[0]
    orders["by_c"] = sel.by_alignment(c, sp.greedy_steps)
    orders["random"] = sel.random_order(len(a), sp.greedy_steps, seed=e)
    return orders, G, c


def method_summary(order: list[int], train_cos: Tensor, transfer: Tensor, ceiling: float, sizes: list[int]) -> dict:
    k_match = next((i + 1 for i, v in enumerate(transfer.tolist()) if v >= ceiling), None)
    k_best = int(transfer.argmax()) + 1
    return {
        "order": order, "cos": train_cos.tolist(), "transfer": transfer.tolist(), "k_matches_full": k_match,
        "best": {"k": k_best, "transfer": float(transfer[k_best - 1]), "cos": float(train_cos[k_best - 1])},
        "at": {k: {"cos": float(train_cos[k - 1]), "transfer": float(transfer[k - 1])} for k in sizes if k <= len(order)},
    }  # fmt: skip


def emotion_result(W, WtW, shifts: ActivationShifts, e: int, sp: SparseSetConfig) -> dict:
    a, a_heldout = shifts.train[e], shifts.heldout[e]
    d = W @ a
    ceiling = float(torch.nn.functional.cosine_similarity(W @ a_heldout, d, dim=0))
    orders, G, c = orders_for(W, WtW, shifts, e, sp)
    methods = {m: method_summary(o, *prefix_curves(W, a, d, a_heldout, o), ceiling, sp.sizes) for m, o in orders.items()}
    taus = {}
    greedy_cos = methods["greedy"]["cos"]
    for tau in sp.taus:
        k0 = next((i + 1 for i, v in enumerate(greedy_cos) if v >= tau), None)
        if k0 is not None:
            shrunk_set = shrink(G, c, orders["greedy"][:k0], tau, sp.max_swaps)
            taus[tau] = {"greedy": set_metrics(W, a, d, a_heldout, orders["greedy"][:k0]).model_dump(),
                         "shrink": set_metrics(W, a, d, a_heldout, shrunk_set).model_dump() | {"neurons": sorted(shrunk_set)}}  # fmt: skip
    return {"ceiling": ceiling, "methods": methods, "tau": taus}


def log_line(emotion: str, r: dict, seconds: float) -> str:
    ms = r["methods"]
    return (f"[{emotion}] full-set transfer {r['ceiling']:.2f} | k to match it: " + ", ".join(f"{m} {v['k_matches_full']}" for m, v in ms.items())
            + " | best: " + ", ".join(f"{m} {v['best']['transfer']:.2f}@{v['best']['k']}" for m, v in ms.items())
            + " | tau: " + ", ".join(f"{t} {v['greedy']['k']}->{v['shrink']['k']}" for t, v in r["tau"].items())
            + f" | {seconds:.0f}s")  # fmt: skip


def main():
    cfg = load_run_config(sys.argv[1:])
    sp = cfg.sparse_set
    t0 = time.time()
    vectors = EmotionVectors.load(cfg.results_dir / cfg.neuron_set.vectors_run / "vectors.safetensors")
    shifts = activation_shifts(cfg.runs_dir / cfg.neuron_set.vectors_run, sp.seed)
    gap = float((shifts.train - vectors.activation_shift()).abs().max())
    assert gap < 1e-3, f"train dA from the pooled sums does not match Stage 2's vectors ({gap})"
    W = target_weights(load_w_down(cfg.model, cfg.layer), vectors, sp.target)
    WtW = W.T @ W
    print(f"[load] W {tuple(W.shape)}, target {sp.target}, {time.time() - t0:.0f}s")
    out = {"config": sp.model_dump(), "emotions": {}}
    for emotion in [e for e in EMOTIONS if not sp.emotions or e in sp.emotions]:
        te = time.time()
        out["emotions"][emotion] = r = emotion_result(W, WtW, shifts, EMOTIONS.index(emotion), sp)
        print(log_line(emotion, r, time.time() - te), flush=True)
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "sparse_set.json").write_text(json.dumps(out))
    print(f"[saved] {cfg.result_dir}/sparse_set.json")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
