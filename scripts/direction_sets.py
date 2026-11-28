"""The sparsest set of earlier MLP neurons behind a direction, and its classification accuracy without them. GPU.

    python scripts/direction_sets.py --config experiments/refusal-sparse-set/refusal_sparse_set.yaml

- See `emotion_neurons.direction_sets`.
- Writes to results/<run_id>/ (the config's results_dir):
  - results.json: the MLP share of the direction, per greedy order its set sizes and top neurons, every curve;
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

from emotion_neurons.config import DirectionSetRunConfig, load_run_config
from emotion_neurons.datasets import example_set
from emotion_neurons.direction_sets import (
    Ordering,
    Parts,
    Vector,
    capture_site,
    curve,
    curve_sizes,
    neuron_order,
    neuron_parts,
    rebuilt_share,
    unit_name,
    w_down,
)
from emotion_neurons.models import adapter_for
from emotion_neurons.sparse import first_reaching

torch.set_grad_enabled(False)


def greedy_summary(ordering: Ordering, parts: Parts, direction: Vector, d_mlp: int) -> dict:
    run = ordering.greedy
    r_hat = direction / direction.norm()
    top = [
        unit_name(int(j), d_mlp) | {"projection": float(parts[:, j].double().cpu() @ r_hat), "norm": float(parts[:, j].norm())}
        for j in run.order[:100]
    ]
    return {
        "size": len(run.order),
        "rebuilt": float(run.rebuilt[-1]),
        "size_at_pct": {str(p): first_reaching(run.rebuilt, p / 100) for p in (50, 75, 90, 95)},
        "top": top,
    }


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
        site = capture_site(model, adapter, rendered, cfg.layer, cfg.position, tokenizer.pad_token_id, cfg.batch_size)
        return site, torch.tensor([e.category == cfg.target for e in chosen])

    train, train_labels = capture("train")
    val, val_labels = capture("val")
    W = w_down(model, adapter, cfg.layer)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[capture] {len(train_labels)} train, {len(val_labels)} val prompts | {time.time() - t0:.0f}s", flush=True)

    direction = train.resid[train_labels].mean(0) - train.resid[~train_labels].mean(0)
    parts = neuron_parts(train.acts[train_labels], train.acts[~train_labels], W).to(cfg.device)
    del W
    d_mlp = train.acts.shape[-1]
    mlp_write = parts.double().sum(1).cpu()
    data = dict(direction=direction, train=(train.resid, train_labels), val=(val.resid, val_labels))

    points, greedy, tensors = [], {}, {"direction": direction.float().contiguous()}
    for spec in cfg.orders:
        ordering = neuron_order(spec, parts, direction)
        print(f"[{ordering.name}] {len(ordering.order)} neurons | {time.time() - t0:.0f}s", flush=True)
        tensors[f"{ordering.name}/order"] = ordering.order.contiguous()
        if ordering.greedy is not None:
            greedy[ordering.name] = greedy_summary(ordering, parts, direction, d_mlp)
            tensors[f"{ordering.name}/rebuilt"] = ordering.greedy.rebuilt.contiguous()
        for p in curve(ordering.name, parts, ordering.order, curve_sizes(cfg.sizes, len(ordering.order)), **data):
            print(f"[{p.order} {p.removed}] rebuilt {p.rebuilt:.3f}, cosine {p.cosine_with_direction:.3f}, val accuracy {p.val_accuracy:.3f}")  # fmt: skip
            points.append(p)

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "model": cfg.model.name,
        "site": {"layer": cfg.layer, "position": cfg.position},
        "target": cfg.target,
        "n_train": len(train_labels),
        "n_val": len(val_labels),
        "pool": {"layers": cfg.layer, "d_mlp": d_mlp, "neurons": parts.shape[1]},
        "direction_norm": float(direction.norm()),
        "mlp_write": {
            "cosine_with_direction": float(F.cosine_similarity(mlp_write, direction, dim=0)),
            "projection_share": float(mlp_write @ direction / (direction @ direction)),
            "rebuilt": rebuilt_share(direction, mlp_write),
        },
        "greedy": greedy,
        "curve": [p.model_dump() for p in points],
        "seconds": round(time.time() - t0),
    }
    (cfg.result_dir / "results.json").write_text(json.dumps(out, indent=1))
    save_file(tensors, str(cfg.result_dir / "orders.safetensors"))
    print(f"[saved] {cfg.result_dir}/results.json, orders.safetensors | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
