"""Loss and KL of a model whose MLP neurons or attention writers are mean-ablated between their poles, against the
intact model. GPU.

    python scripts/ablation.py --config experiments/goodfire-sparsity/ablation_full.yaml

- See `emotion_neurons.ablation`. Needs a poles.py run of the same model, for every ablated layer.
- Text: held-out general text, loss and KL at every position. Prompts (optional): KL at the post-instruction
  positions of the val split, per category.
- Writes results/<run_id>/ablation.json: one `AblationResult` per configured ablation.
"""

import json
import sys
import time

import torch

from emotion_neurons.ablation import (
    MeanAblation,
    ablation_result,
    compare,
    load_poles,
    prompt_batches,
    prompt_kl,
    text_batches,
)
from emotion_neurons.config import AblationRunConfig, load_run_config
from emotion_neurons.datasets import example_set
from emotion_neurons.models import adapter_for
from emotion_neurons.poles import collect_windows

torch.set_grad_enabled(False)


def main():
    cfg = load_run_config(AblationRunConfig, sys.argv[1:])
    t0 = time.time()
    torch.manual_seed(cfg.mask_seed)
    adapter = adapter_for(cfg.model)
    model, tokenizer = adapter.load(cfg.device)
    n_layers = model.config.num_hidden_layers
    poles = load_poles(cfg.thresholds, cfg.low, cfg.high, model.device)
    ablations = [MeanAblation(poles, a.kind, a.layers(n_layers), a.positions, cfg.writer) for a in cfg.ablations]
    names = [a.name(n_layers) for a in cfg.ablations]

    windows = collect_windows(cfg.text, tokenizer)
    batches = text_batches(windows, tokenizer.pad_token_id, cfg.text.batch_size, cfg.text.skip_first, model.device)
    text = compare(model, adapter, batches, ablations)
    results = [ablation_result(n, a, a.ablated_fraction(), t) for n, a, t in zip(names, ablations, text)]
    for r in results:
        print(
            f"[text] {r.name}: ablated {r.ablated_fraction:.3f}, loss {r.loss_intact:.4f} -> {r.loss_ablated:.4f}, KL {r.kl_per_position:.4f}"
        )

    if cfg.prompts:
        examples = example_set(cfg.prompts)
        for category in examples.categories:
            val = [
                examples.render(e, tokenizer, adapter) for e in examples.examples() if e.category == category and e.split == "val"
            ]
            rows, positions = [r.input_ids for r in val], [r.positions for r in val]
            batches = prompt_batches(rows, positions, tokenizer.pad_token_id, cfg.prompt_batch_size, model.device)
            for r, t in zip(results, compare(model, adapter, batches, ablations)):
                r.prompts[category] = prompt_kl(t)
                print(f"[prompts] {r.name} {category}: KL {r.prompts[category].kl_per_position:.4f}")

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "model": cfg.model.name,
        "thresholds": str(cfg.thresholds),
        "band": [cfg.low, cfg.high],
        "writer": cfg.writer.model_dump(),
        "ablations": [r.model_dump() for r in results],
    }
    (cfg.result_dir / "ablation.json").write_text(json.dumps(out, indent=1))
    print(f"[saved] {cfg.result_dir}/ablation.json")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
