"""A difference-of-means direction, selected and checked as in the refusal paper (`emotion_neurons.directions`).

- capture (GPU): one pass over the train examples; resid_pre at every layer and rendered position, mean per category.
  - runs/<run_id>/means.safetensors: [position, layer, d_model] per category, fp64.
- build (CPU): the difference of means for every position and layer, the candidates.
  - results/<run_id>/direction.safetensors: `diffs` [position, layer, d_model] and `means`, fp32.
- select (GPU): each candidate scored on the val prompts; the kept one with the lowest bypass score wins.
  - results/<run_id>/direction.json: the grid and the choice; `direction` added to direction.safetensors.
- check (GPU): greedy replies, plain and with the direction ablated or added.
  - results/<run_id>/replies.json; `refusal_rates` added to direction.json.
"""

import json
import time

import torch
from safetensors.torch import load_file, save_file
from transformers import PreTrainedTokenizerBase

from ..config import ExtractionRunConfig
from ..datasets import example_set
from ..directions import category_sums, diff_from_means, generation_check, refusal_token_ids, select_direction
from ..models import adapter_for
from .loaded import loaded_model


def target_and_other(cfg: ExtractionRunConfig) -> tuple[str, str]:
    target = cfg.direction.diff_means.target
    return target, next(c for c in cfg.data.categories if c != target)


def val_prompts(cfg: ExtractionRunConfig, tokenizer: PreTrainedTokenizerBase) -> dict[str, list[list[int]]]:
    examples_of = example_set(cfg.data)
    adapter = adapter_for(cfg.model)
    examples = [e for e in examples_of.examples() if e.split == "val"]
    return {
        c: [examples_of.render(e, tokenizer, adapter).input_ids for e in examples if e.category == c] for c in cfg.data.categories
    }


def capture(cfg: ExtractionRunConfig) -> None:
    t0 = time.time()
    model, tokenizer = loaded_model(cfg.model, cfg.device)
    examples_of = example_set(cfg.data)
    train = [e for e in examples_of.examples() if e.split == "train"]
    batch_size = cfg.direction.selection.batch_size
    sums = category_sums(model, tokenizer, adapter_for(cfg.model), train, examples_of, batch_size)
    means = {c: (sums[c][0] / sums[c][1]).contiguous() for c in cfg.data.categories}
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    save_file(means, str(cfg.run_dir / "means.safetensors"))
    shape = next(iter(means.values())).shape
    print(f"[means] {len(train)} train prompts, {shape[0]} positions x {shape[1]} layers | {time.time() - t0:.0f}s", flush=True)


def build(cfg: ExtractionRunConfig) -> None:
    categories = list(cfg.data.categories)
    saved = load_file(str(cfg.run_dir / "means.safetensors"))
    means = torch.stack([saved[c] for c in categories])  # [category, position, layer, d]
    diffs = diff_from_means(means, categories, cfg.direction.diff_means)
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    tensors = {"diffs": diffs.float().contiguous(), "means": means.float().contiguous()}
    save_file(tensors, str(cfg.result_dir / "direction.safetensors"))
    print(f"[saved] {cfg.result_dir}/direction.safetensors: {diffs.shape[0]} positions x {diffs.shape[1]} layers")


def select(cfg: ExtractionRunConfig) -> None:
    t0 = time.time()
    model, tokenizer = loaded_model(cfg.model, cfg.device)
    tensors = load_file(str(cfg.result_dir / "direction.safetensors"))
    diffs = tensors["diffs"].double()
    target, other = target_and_other(cfg)
    s = cfg.direction.selection
    refusal_ids = refusal_token_ids(tokenizer, s.refusal_tokens)
    val = val_prompts(cfg, tokenizer)
    selection = select_direction(model, s, diffs, val, target, other, refusal_ids, tokenizer.pad_token_id)
    choice = selection["choice"]
    vector = diffs[choice["position"], choice["layer"]].to(model.dtype)
    print(
        f"[select] chose position {choice['position']}, layer {choice['layer']}: {choice} | {time.time() - t0:.0f}s", flush=True
    )

    save_file({**tensors, "direction": vector.float().cpu().contiguous()}, str(cfg.result_dir / "direction.safetensors"))
    d, data = cfg.direction.diff_means, cfg.instructions
    summary = {
        "model": cfg.model.name, "categories": list(data.categories), "target": d.target, "baseline": d.baseline,
        "n_train": data.n_train, "n_val": data.n_val, "choice": choice,
        "filtered_out_all": selection["filtered_out_all"], "grid": selection["grid"],
    }  # fmt: skip
    (cfg.result_dir / "direction.json").write_text(json.dumps(summary, indent=1))
    print(f"[saved] {cfg.result_dir}/direction.json", flush=True)


def check(cfg: ExtractionRunConfig) -> None:
    model, tokenizer = loaded_model(cfg.model, cfg.device)
    summary = json.loads((cfg.result_dir / "direction.json").read_text())
    vector = load_file(str(cfg.result_dir / "direction.safetensors"))["direction"].to(model.device, model.dtype)
    target, other = target_and_other(cfg)
    val = val_prompts(cfg, tokenizer)
    result = generation_check(model, tokenizer, cfg.direction.generation, val, target, other, vector, summary["choice"]["layer"])
    summary["refusal_rates"] = {k: v["refusal_rate"] for k, v in result.items()}
    (cfg.result_dir / "direction.json").write_text(json.dumps(summary, indent=1))
    (cfg.result_dir / "replies.json").write_text(json.dumps({k: v["replies"] for k, v in result.items()}, indent=1))
    print(f"[saved] {cfg.result_dir}/replies.json, refusal rates {summary['refusal_rates']}", flush=True)


STEPS = {"capture": capture, "build": build, "select": select, "check": check}
