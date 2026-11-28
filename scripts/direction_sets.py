"""The sparsest set of MLP neurons behind a direction, and its classification accuracy without them. GPU.

    python scripts/direction_sets.py --config experiments/refusal-sparse-set/refusal_sparse_set.yaml

- See `emotion_neurons.direction_sets`.
- Writes to results/<run_id>/ (the config's results_dir):
  - results.json: the pool's share of the direction, per greedy order its set sizes and top neurons,
    the random-direction baseline, and per order and evaluation the accuracy curve;
  - orders.safetensors: the direction, and per order `<name>/order` (layer * d_mlp + neuron);
    greedy orders also `<name>/rebuilt`, the share of the direction rebuilt per step.
"""

import gc
import json
import sys
import time

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import Tensor

from emotion_neurons.config import AblationEvaluation, DirectionSetRunConfig, Evaluation, load_run_config
from emotion_neurons.datasets import example_set
from emotion_neurons.direction_sets import (
    Ordering,
    Parts,
    Pool,
    Vector,
    band_subsets,
    capture_site,
    curve,
    curve_sizes,
    flat_acts,
    flat_w,
    neuron_order,
    neuron_parts,
    neuron_ranks,
    pool_means,
    pool_poles,
    random_baseline,
    rebuilt_share,
    site_vectors,
    tail_rates,
    unit_name,
    w_down,
)
from emotion_neurons.models import adapter_for
from emotion_neurons.sparse import first_reaching

torch.set_grad_enabled(False)


def greedy_summary(ordering: Ordering, parts: Parts, direction: Vector, layers: list[int], d_mlp: int) -> dict:
    run = ordering.greedy
    r_hat = direction / direction.norm()
    top = [
        unit_name(int(j), layers, d_mlp)
        | {"projection": float(parts[:, j].double().cpu() @ r_hat), "norm": float(parts[:, j].norm())}
        for j in run.order[:100]
    ]
    return {
        "size": len(run.order),
        "rebuilt": float(run.rebuilt[-1]),
        "size_at_pct": {str(p): first_reaching(run.rebuilt, p / 100) for p in (50, 75, 90, 95)},
        "top": top,
    }


def means_for(evaluations: list[Evaluation], layers: list[int], device: str) -> Tensor | None:
    paths = {e.means for e in evaluations if isinstance(e, AblationEvaluation)}
    assert len(paths) <= 1, f"one means file per run, got {paths}"
    return pool_means(paths.pop(), layers).to(device) if paths else None


def main():
    cfg = load_run_config(DirectionSetRunConfig, sys.argv[1:])
    t0 = time.time()
    adapter = adapter_for(cfg.model)
    model, tokenizer = adapter.load(cfg.device)
    prompts = example_set(cfg.data)
    examples = prompts.examples()

    def capture(split: str):
        chosen = [e for e in examples if e.split == split]
        rendered = [prompts.render(e, tokenizer, adapter) for e in chosen]
        site = capture_site(model, adapter, rendered, cfg.site, cfg.position, tokenizer.pad_token_id, cfg.batch_size)
        return site, torch.tensor([e.category == cfg.target for e in chosen])

    train, train_labels = capture("train")
    val, val_labels = capture("val")
    layers = cfg.site.pool_layers
    W = w_down(model, adapter, layers)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[capture] {len(train_labels)} train, {len(val_labels)} val prompts | {time.time() - t0:.0f}s", flush=True)

    train_vectors, val_vectors = site_vectors(cfg.site, train, W), site_vectors(cfg.site, val, W)
    direction = train_vectors[train_labels].mean(0) - train_vectors[~train_labels].mean(0)
    parts = neuron_parts(train.acts[train_labels], train.acts[~train_labels], W).to(cfg.device)
    pool = Pool(
        parts=parts,
        W=flat_w(W).to(cfg.device),
        train_acts=flat_acts(train.acts).to(cfg.device),
        val_acts=flat_acts(val.acts).to(cfg.device),
        means=means_for(cfg.evaluations, layers, cfg.device),
    )
    del W
    d_mlp = train.acts.shape[-1]
    pool_write = parts.double().sum(1).cpu()
    data = dict(direction=direction, train=(train_vectors, train_labels), val=(val_vectors, val_labels))

    points, greedy, orders, tensors = [], {}, {}, {"direction": direction.float().contiguous()}
    for spec in cfg.orders:
        ordering = neuron_order(spec, parts, direction)
        orders[ordering.name] = ordering.order
        print(f"[{ordering.name}] {len(ordering.order)} neurons | {time.time() - t0:.0f}s", flush=True)
        tensors[f"{ordering.name}/order"] = ordering.order.contiguous()
        if ordering.greedy is not None:
            greedy[ordering.name] = greedy_summary(ordering, parts, direction, layers, d_mlp)
            tensors[f"{ordering.name}/rebuilt"] = ordering.greedy.rebuilt.contiguous()
        sizes = curve_sizes(cfg.sizes, len(ordering.order))
        for p in curve(ordering.name, pool, ordering.order, sizes, cfg.evaluations, **data):
            print(f"[{p.order} {p.evaluation} {p.neurons}] rebuilt {p.rebuilt:.3f}, cosine {p.cosine_with_direction:.3f}, val accuracy {p.val_accuracy:.3f}")  # fmt: skip
            points.append(p)

    baseline = None
    if cfg.random_directions is not None:
        baseline = random_baseline(cfg.random_directions, data["train"], data["val"])
        print(f"[random directions] {baseline.count}, val accuracy mean {baseline.mean:.3f}, quantiles {baseline.quantiles}")

    subsets = None
    if cfg.band_subsets is not None:
        spec = cfg.band_subsets
        assert "random_seed0" in orders, "band subsets start from the random_seed0 order"
        low, high = pool_poles(spec.thresholds, spec.low, spec.high, layers)
        rates = tail_rates(flat_acts(train.acts), low, high)
        assert "greedy" in orders, "band subsets rank neurons by the greedy order"
        ranks = neuron_ranks(parts, direction, orders["greedy"])
        first = orders["random_seed0"][: spec.set_size]
        subsets = band_subsets(spec, first, rates, ranks, pool, direction, data["train"], data["val"])
        for i, b in enumerate(subsets):
            drops = sum(b.drop_accuracies) / len(b.drop_accuracies) if b.drop_accuracies else None
            print(f"[band set {i}] in band {len(b.in_band)}/{len(b.neurons)}, all {b.all_accuracy:.3f}, in band {b.in_band_accuracy}, tail {b.tail_accuracy}, random drop mean {drops}")  # fmt: skip

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "model": cfg.model.name,
        "site": cfg.site.model_dump() | {"position": cfg.position},
        "target": cfg.target,
        "n_train": len(train_labels),
        "n_val": len(val_labels),
        "pool": {"layers": layers, "d_mlp": d_mlp, "neurons": parts.shape[1]},
        "direction_norm": float(direction.norm()),
        "pool_write": {
            "cosine_with_direction": float(F.cosine_similarity(pool_write, direction, dim=0)),
            "projection_share": float(pool_write @ direction / (direction @ direction)),
            "rebuilt": rebuilt_share(direction, pool_write),
        },
        "greedy": greedy,
        "random_directions": baseline.model_dump() if baseline else None,
        "band_subsets": [b.model_dump() for b in subsets] if subsets else None,
        "curve": [p.model_dump() for p in points],
        "seconds": round(time.time() - t0),
    }
    (cfg.result_dir / "results.json").write_text(json.dumps(out, indent=1))
    save_file(tensors, str(cfg.result_dir / "orders.safetensors"))
    print(f"[saved] {cfg.result_dir}/results.json, orders.safetensors | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
