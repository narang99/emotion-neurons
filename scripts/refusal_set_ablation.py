"""The refusal direction and the model's own refusal rate with the first k neurons of each order mean-ablated. GPU.

    python scripts/refusal_set_ablation.py --config experiments/refusal-set-ablation/refusal_set_ablation.yaml

- Needs a direction_sets.py run whose orders cover layers 0 to layer - 1 (`orders`).
- See `ablation.SetAblation` and `refusal_ablation.run_condition`.
- Writes to results/<run_id>/ (the config's results_dir):
  - results.json: the intact direction, one record per order and size: cosine, norm, refusal rates;
  - replies.json: the val replies per condition and category.
"""

import json
import sys
import time

import torch
from safetensors.torch import load_file

from emotion_neurons.ablation import set_ablation
from emotion_neurons.config import RefusalSetAblationRunConfig, load_run_config
from emotion_neurons.datasets import example_set
from emotion_neurons.models import adapter_for
from emotion_neurons.refusal_ablation import Condition, condition_result, run_condition, site_vectors

torch.set_grad_enabled(False)


def main():
    cfg = load_run_config(RefusalSetAblationRunConfig, sys.argv[1:])
    t0 = time.time()
    adapter = adapter_for(cfg.model)
    model, tokenizer = adapter.load(cfg.device)
    prompts = example_set(cfg.data)
    examples = prompts.examples()
    train = {
        c: [prompts.render(e, tokenizer, adapter) for e in examples if e.category == c and e.split == "train"]
        for c in prompts.categories
    }
    val = {
        c: [prompts.render(e, tokenizer, adapter).input_ids for e in examples if e.category == c and e.split == "val"]
        for c in prompts.categories
    }
    phrases = cfg.refusal_check.refusal_phrases
    run = dict(train=train, val=val, layer=cfg.layer, position=cfg.position, batch_size=cfg.batch_size, g=cfg.refusal_check)
    orders = load_file(str(cfg.orders))
    poles = load_file(str(cfg.means))
    mean_layers = poles["layers"].tolist()
    d_mlp = poles["mean"].shape[1]

    acts, intact_replies, _ = run_condition(model, tokenizer, adapter, ablation=None, **run)
    intact_vectors = site_vectors(acts, cfg.target)
    intact = intact_vectors["direction"]
    replies = {"intact": intact_replies}
    results = [condition_result(Condition(name="intact"), intact_vectors, intact, intact_replies, None, phrases)]
    print(f"[intact] norm {results[0].norm:.2f}, refusal {results[0].refusal_rates}", flush=True)

    for name in cfg.order_names:
        order = orders[f"{name}/order"]
        for size in cfg.sizes:
            assert size <= len(order), f"{name} has {len(order)} neurons, {size} wanted"
            ablation = set_ablation(order[:size], poles["mean"], mean_layers, d_mlp)
            condition = Condition(name=f"{name}/k{size}", kind=name, layers=ablation.layers)
            condition_acts, condition_replies, fraction = run_condition(model, tokenizer, adapter, ablation=ablation, **run)
            result = condition_result(
                condition, site_vectors(condition_acts, cfg.target), intact, condition_replies, fraction, phrases
            )
            replies[condition.name] = condition_replies
            results.append(result)
            print(
                f"[{condition.name}] cosine {result.cosine:.4f}, norm {result.norm:.2f}, refusal {result.refusal_rates}, {time.time() - t0:.0f}s",
                flush=True,
            )

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "model": cfg.model.name,
        "site": {"layer": cfg.layer, "position": cfg.position},
        "target": cfg.target,
        "n_val": {c: len(r) for c, r in val.items()},
        "conditions": [r.model_dump() for r in results],
        "seconds": round(time.time() - t0),
    }
    (cfg.result_dir / "results.json").write_text(json.dumps(out, indent=1))
    (cfg.result_dir / "replies.json").write_text(json.dumps(replies, indent=1))
    print(f"[saved] {cfg.result_dir}/results.json, replies.json | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
