"""A difference-of-means direction, selected and checked as in the refusal paper (`emotion_neurons.directions`).

- capture (GPU): one pass over the train examples; resid_pre at every layer and rendered position, mean per category.
  - runs/<run_id>/means.safetensors: [position, layer, d_model] per category, fp64.
- build (CPU): the difference of means for every position and layer, the candidates.
  - results/<run_id>/direction.safetensors: `diffs` [position, layer, d_model] and `means`, fp32.
- select (GPU): each candidate scored on the val prompts; the kept one with the lowest bypass score wins.
  - results/<run_id>/direction.json: the grid and the choice; `direction` added to direction.safetensors.
- check (GPU): greedy replies, plain and with the direction ablated or added.
  - results/<run_id>/replies.json; `refusal_rates` added to direction.json.
- tokens (GPU) and report (CPU) write token_scores.* and report.html.
"""

import json
import time
from functools import cached_property

import torch
from safetensors.torch import load_file, save_file
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..config import DirectionExtraction, ExtractionRunConfig
from ..datasets import example_set
from ..directions import category_sums, diff_from_means, generation_check, refusal_token_ids, select_direction
from ..models import ModelAdapter, adapter_for
from ..reports import build_direction_report, write_report
from ..token_scores import direction_token_scores, save_token_scores


def target_and_other(cfg: ExtractionRunConfig) -> tuple[str, str]:
    target = cfg.direction.target
    return target, next(c for c in cfg.data.categories if c != target)


def val_prompts(
    cfg: ExtractionRunConfig, tokenizer: PreTrainedTokenizerBase, adapter: ModelAdapter
) -> dict[str, list[list[int]]]:
    examples_of = example_set(cfg.data)
    examples = [e for e in examples_of.examples() if e.split == "val"]
    return {
        c: [examples_of.render(e, tokenizer, adapter).input_ids for e in examples if e.category == c] for c in cfg.data.categories
    }


class DirectionMethod:
    def __init__(self, cfg: ExtractionRunConfig):
        assert isinstance(cfg.extraction, DirectionExtraction), "not a direction run"
        self.cfg = cfg
        self.adapter = adapter_for(cfg.model)

    @cached_property
    def loaded(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Loaded by the first step that needs the model, then shared by the later steps."""
        return self.adapter.load(self.cfg.device)

    def capture(self) -> None:
        t0 = time.time()
        model, tokenizer = self.loaded
        examples_of = example_set(self.cfg.data)
        train = [e for e in examples_of.examples() if e.split == "train"]
        batch_size = self.cfg.direction.selection.batch_size
        sums = category_sums(model, tokenizer, self.adapter, train, examples_of, batch_size)
        means = {c: (sums[c][0] / sums[c][1]).contiguous() for c in self.cfg.data.categories}
        self.cfg.run_dir.mkdir(parents=True, exist_ok=True)
        save_file(means, str(self.cfg.run_dir / "means.safetensors"))
        shape = next(iter(means.values())).shape
        print(
            f"[means] {len(train)} train prompts, {shape[0]} positions x {shape[1]} layers | {time.time() - t0:.0f}s", flush=True
        )

    def build(self) -> None:
        categories = list(self.cfg.data.categories)
        saved = load_file(str(self.cfg.run_dir / "means.safetensors"))
        means = torch.stack([saved[c] for c in categories])  # [category, position, layer, d]
        diffs = diff_from_means(means, categories, self.cfg.direction.target)
        self.cfg.result_dir.mkdir(parents=True, exist_ok=True)
        tensors = {"diffs": diffs.float().contiguous(), "means": means.float().contiguous()}
        save_file(tensors, str(self.cfg.result_dir / "direction.safetensors"))
        print(f"[saved] {self.cfg.result_dir}/direction.safetensors: {diffs.shape[0]} positions x {diffs.shape[1]} layers")

    def select(self) -> None:
        t0 = time.time()
        model, tokenizer = self.loaded
        tensors = load_file(str(self.cfg.result_dir / "direction.safetensors"))
        diffs = tensors["diffs"].double()
        target, other = target_and_other(self.cfg)
        s = self.cfg.direction.selection
        refusal_ids = refusal_token_ids(tokenizer, s.refusal_tokens)
        val = val_prompts(self.cfg, tokenizer, self.adapter)
        selection = select_direction(model, s, diffs, val, target, other, refusal_ids, tokenizer.pad_token_id)
        choice = selection["choice"]
        vector = diffs[choice["position"], choice["layer"]].to(model.dtype)
        print(
            f"[select] chose position {choice['position']}, layer {choice['layer']}: {choice} | {time.time() - t0:.0f}s",
            flush=True,
        )

        save_file({**tensors, "direction": vector.float().cpu().contiguous()}, str(self.cfg.result_dir / "direction.safetensors"))
        data = self.cfg.instructions
        summary = {
            "model": self.cfg.model.name, "categories": list(data.categories), "target": self.cfg.direction.target,
            "n_train": data.n_train, "n_val": data.n_val, "choice": choice,
            "filtered_out_all": selection["filtered_out_all"], "grid": selection["grid"],
        }  # fmt: skip
        (self.cfg.result_dir / "direction.json").write_text(json.dumps(summary, indent=1))
        print(f"[saved] {self.cfg.result_dir}/direction.json", flush=True)

    def check(self) -> None:
        model, tokenizer = self.loaded
        summary = json.loads((self.cfg.result_dir / "direction.json").read_text())
        vector = load_file(str(self.cfg.result_dir / "direction.safetensors"))["direction"].to(model.device, model.dtype)
        target, other = target_and_other(self.cfg)
        val = val_prompts(self.cfg, tokenizer, self.adapter)
        result = generation_check(
            model, tokenizer, self.cfg.direction.refusal_check, val, target, other, vector, summary["choice"]["layer"]
        )
        summary["refusal_rates"] = {k: v["refusal_rate"] for k, v in result.items()}
        (self.cfg.result_dir / "direction.json").write_text(json.dumps(summary, indent=1))
        (self.cfg.result_dir / "replies.json").write_text(json.dumps({k: v["replies"] for k, v in result.items()}, indent=1))
        print(f"[saved] {self.cfg.result_dir}/replies.json, refusal rates {summary['refusal_rates']}", flush=True)

    def tokens(self) -> None:
        scores, values = direction_token_scores(self.cfg, *self.loaded, self.adapter)
        save_token_scores(self.cfg.result_dir, scores, values)
        print(f"[saved] {self.cfg.result_dir}/token_scores.*: {len(scores.texts)} texts, {values.shape[0]} tokens")

    def report(self) -> None:
        path = write_report("direction.html", build_direction_report(self.cfg), self.cfg.result_dir / "report.html")
        print(f"[saved] {path}")
