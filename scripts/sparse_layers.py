"""Sparse neuron sets for every layer's MLP emotion write. CPU, no forward pass.

    python scripts/sparse_layers.py --config configs/sparse_layers.yaml

- Per layer and emotion: plain greedy on the write (`emotion_neurons.decomposition.sparse`), as at layer 25.
  - The set size is where its held-out transfer matches the full MLP's (`k_match`).
  - Also the smallest prefix with train cosine >= `target_cos`, and the best transfer.
- Activation shifts come from the all-layer pooled sums; W_down per layer is read from the Hub and cached as bf16.
- Writes results/<run_id>/:
  - sparse_layers.json: per layer and emotion, the numbers above and the matching set (greedy order up to k_match);
  - sparse_layers.md and charts/*.svg.
"""

import json
import sys
import time

import torch

from emotion_neurons.capture import LayerMeans
from emotion_neurons.config import RunConfig, load_run_config
from emotion_neurons.decomposition import ActivationShifts, greedy, prefix_curves, shifts_from_means
from emotion_neurons.decomposition import selection as sel
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.journey import contrast
from emotion_neurons.models import load_cached_tensor
from emotion_neurons.svg import line_chart
from emotion_neurons.texts import StorySplit, load_story_records

torch.set_grad_enabled(False)


def emotion_set(W, WtW, shifts: ActivationShifts, e: int, steps: int, target_cos: float) -> dict:
    a, a_heldout = shifts.train[e], shifts.heldout[e]
    d = W @ a
    ceiling = float(torch.nn.functional.cosine_similarity(W @ a_heldout, d, dim=0))
    order, _ = greedy(sel.gram(WtW, a), sel.plain(W, a), steps)
    train_cos, transfer = prefix_curves(W, a, d, a_heldout, order)
    k_match = next((i + 1 for i, v in enumerate(transfer.tolist()) if v >= ceiling), None)
    k_target = next((i + 1 for i, v in enumerate(train_cos.tolist()) if v >= target_cos), None)
    k_best = int(transfer.argmax()) + 1
    return {
        "ceiling": ceiling,
        "k_match": k_match,
        "cos_at_match": float(train_cos[k_match - 1]) if k_match else None,
        "k_target_cos": k_target,
        "best_transfer": float(transfer[k_best - 1]),
        "k_best": k_best,
        "neurons": order[:k_match] if k_match else None,
    }


def layer_sets(cfg: RunConfig, means: LayerMeans, split: StorySplit, layer: int) -> dict:
    sl = cfg.sparse_layers
    shifts = shifts_from_means(means.site(layer, "mlp_act").float(), split, sl.seed)
    W = load_cached_tensor(cfg.model, f"model.layers.{layer}.mlp.down_proj.weight")
    write = contrast(means.site(layer, "mlp_out"), split).float()
    rebuild = float(((shifts.train @ W.T - write).norm() / write.norm()))
    assert rebuild < 0.05, f"layer {layer}: W_down @ dA differs from the MLP write by {rebuild:.3f}"
    WtW = W.T @ W
    sets = {emotion: emotion_set(W, WtW, shifts, e, sl.greedy_steps, sl.target_cos) for e, emotion in enumerate(EMOTIONS)}
    return {"rebuild_rel_err": rebuild, "emotions": sets}


def mean_of(values: list[float]) -> str:
    return f"{sum(values) / len(values):.2f}" if values else "-"


def summary_rows(results: dict[int, dict]) -> list[str]:
    rows = [
        "| layer | k_match (median, min-max) | cos at match | k for cos target | ceiling | best transfer |",
        "|---|---|---|---|---|---|",
    ]
    for layer, r in results.items():
        sets = r["emotions"].values()
        ks = sorted(s["k_match"] for s in sets if s["k_match"])
        targets = sorted(s["k_target_cos"] for s in sets if s["k_target_cos"])
        median = ks[len(ks) // 2] if ks else None
        rows.append(
            f"| {layer} | {median} ({ks[0] if ks else '-'}-{ks[-1] if ks else '-'}; {len(ks)}/12 matched) | "
            f"{mean_of([s['cos_at_match'] for s in sets if s['k_match']])} | {targets[len(targets) // 2] if targets else '-'} | "
            f"{sum(s['ceiling'] for s in sets) / 12:.2f} | {sum(s['best_transfer'] for s in sets) / 12:.2f} |"
        )
    return rows


def main():
    cfg = load_run_config(sys.argv[1:])
    t0 = time.time()
    run_dir = cfg.runs_dir / cfg.sparse_layers.source_run
    means = LayerMeans(run_dir)
    split = StorySplit.of(load_story_records(run_dir))
    layers = cfg.sparse_layers.layers or list(range(means.n_layers))
    results = {}
    for layer in layers:
        tl = time.time()
        results[layer] = r = layer_sets(cfg, means, split, layer)
        ks = [s["k_match"] for s in r["emotions"].values()]
        print(f"[layer {layer:2d}] rebuild {r['rebuild_rel_err']:.4f} | k_match {ks} | {time.time() - tl:.0f}s", flush=True)

    out = cfg.result_dir
    (out / "charts").mkdir(parents=True, exist_ok=True)
    (out / "sparse_layers.json").write_text(json.dumps({"config": cfg.sparse_layers.model_dump(), "layers": results}))
    lines = ["# Sparse sets of every layer's MLP emotion write", "", f"- greedy steps {cfg.sparse_layers.greedy_steps}; "
             f"k_match: smallest prefix whose held-out transfer reaches the full MLP's (the ceiling).", "", *summary_rows(results)]  # fmt: skip
    (out / "sparse_layers.md").write_text("\n".join(lines) + "\n")
    series = {
        emotion: [results[layer]["emotions"][emotion]["k_match"] or cfg.sparse_layers.greedy_steps for layer in layers]
        for emotion in ("happy", "sad", "angry", "afraid", "proud")
    }
    (out / "charts" / "k_match.svg").write_text(line_chart("Neurons needed to match the full MLP on held-out stories", series))
    print(f"[saved] {out}/sparse_layers.json, sparse_layers.md, charts/")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
