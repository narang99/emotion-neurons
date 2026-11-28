"""The refusal direction at one site with the MLPs of earlier layers mean-ablated: cosine, norm, refusal rate. GPU.

    python scripts/refusal_ablation.py --config experiments/refusal-ablation/refusal_ablation.yaml

- See `emotion_neurons.refusal_ablation`. Needs poles.py runs of the same model for every band.
- Writes to results/<run_id>/ (the config's results_dir):
  - directions.safetensors: `<condition>/<category>_mean` and `<condition>/direction`, fp32 [d_model];
  - results.json: the split-half floor, the intact direction's cosine with `reference`, one record per condition;
  - replies.json: the val replies per condition and category.
"""

import json
import sys
import time

import torch
from safetensors.torch import load_file, save_file

from emotion_neurons.ablation import MeanAblation, load_poles
from emotion_neurons.config import RefusalAblationRunConfig, load_run_config
from emotion_neurons.datasets import example_set, examples_in
from emotion_neurons.models import adapter_for
from emotion_neurons.refusal_ablation import (
    Condition,
    condition_result,
    cosine,
    run_condition,
    site_vectors,
    split_half_floor,
)

torch.set_grad_enabled(False)


def main():
    cfg = load_run_config(RefusalAblationRunConfig, sys.argv[1:])
    t0 = time.time()
    adapter = adapter_for(cfg.model)
    model, tokenizer = adapter.load(cfg.device)
    prompts = example_set(cfg.data)
    examples = prompts.examples()
    train, val = {}, {}
    for c in prompts.categories:
        train[c] = [prompts.render(e, tokenizer, adapter) for e in examples_in(examples, "train", c)]
        val[c] = [prompts.render(e, tokenizer, adapter).input_ids for e in examples_in(examples, "val", c)]
    other = next(c for c in prompts.categories if c != cfg.target)
    phrases = cfg.refusal_check.refusal_phrases
    run = dict(train=train, val=val, layer=cfg.layer, position=cfg.position, batch_size=cfg.batch_size, g=cfg.refusal_check)

    acts, intact_replies, _ = run_condition(model, tokenizer, adapter, ablation=None, **run)
    intact_vectors = site_vectors(acts, cfg.target)
    intact = intact_vectors["direction"]
    floor = split_half_floor(acts[cfg.target], acts[other], cfg.split_halves)
    reference = cosine(intact, load_file(str(cfg.reference))["direction"].double()) if cfg.reference else None
    tensors = {f"intact/{k}": v for k, v in intact_vectors.items()}
    replies = {"intact": intact_replies}
    results = [condition_result(Condition(name="intact"), intact_vectors, intact, intact_replies, None, phrases)]
    print(
        f"[intact] norm {results[0].norm:.2f}, refusal {results[0].refusal_rates}, floor mean {floor.mean:.4f} min {floor.min:.4f}, reference cosine {reference}",
        flush=True,
    )

    for band, spec in cfg.bands.items():
        poles = load_poles(spec.thresholds, spec.low, spec.high, model.device)
        for kind in cfg.kinds:
            for start in cfg.starts:
                condition = Condition(
                    name=f"{band}/{kind}/start{start}", band=band, kind=kind, start=start, layers=list(range(start, cfg.layer))
                )
                torch.manual_seed(cfg.mask_seed)
                ablation = MeanAblation(poles, kind, condition.layers)
                condition_acts, condition_replies, fraction = run_condition(model, tokenizer, adapter, ablation=ablation, **run)
                vectors = site_vectors(condition_acts, cfg.target)
                result = condition_result(condition, vectors, intact, condition_replies, fraction, phrases)
                tensors |= {f"{condition.name}/{k}": v for k, v in vectors.items()}
                replies[condition.name] = condition_replies
                results.append(result)
                print(
                    f"[{condition.name}] ablated {fraction:.3f}, cosine {result.cosine:.4f}, norm {result.norm:.2f}, refusal {result.refusal_rates}, {time.time() - t0:.0f}s",
                    flush=True,
                )

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.float().contiguous() for k, v in tensors.items()}, str(cfg.result_dir / "directions.safetensors"))
    out = {
        "model": cfg.model.name,
        "site": {"layer": cfg.layer, "position": cfg.position},
        "target": cfg.target,
        "n_train": {c: len(r) for c, r in train.items()},
        "n_val": {c: len(r) for c, r in val.items()},
        "split_half_floor": floor.model_dump(),
        "reference_cosine": reference,
        "conditions": [r.model_dump() for r in results],
    }
    (cfg.result_dir / "results.json").write_text(json.dumps(out, indent=1))
    (cfg.result_dir / "replies.json").write_text(json.dumps(replies, indent=1))
    print(f"[saved] {cfg.result_dir}/directions.safetensors, results.json, replies.json")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
