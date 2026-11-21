"""A difference-of-means direction on instruction data, selected and checked as in the refusal paper
(Arditi et al. 2024, "Refusal in language models is mediated by a single direction").

1. Means: one pass over the train instructions; the residual stream at every layer's input (resid_pre), at each
   post-instruction position, summed per category. Nothing per example is stored.
2. Candidates: for every layer and post-instruction position, the difference of means (`vectors.diff_means`).
3. Selection, on the val instructions:
   - bypass: the refusal score of target-category prompts with the direction ablated everywhere (lower is better);
   - induce: the refusal score of the other prompts with the direction added at its layer (must be > 0);
   - kl: the next-token KL of the other prompts between the plain model and the ablated one (must be < max_kl);
   - layers in the last 20% are skipped. The kept candidate with the lowest bypass score wins.
   - Refusal score: log-odds that the reply starts with one of `refusal_tokens` ("I", "As" for Qwen).
4. Generation check: greedy replies of the val prompts, counted as refusals by phrase matching, plain and with
   the direction ablated (target prompts) or added (other prompts).
- Directional ablation removes the direction from the residual stream at every layer's input and from every
  attention and MLP output, so it never re-enters. Addition adds the raw difference of means at one layer's input.
"""

import json
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from jaxtyping import Float
from safetensors.torch import save_file
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import DirectionRunConfig
from .datasets import Example, InstructionSet
from .models import ModelAdapter, adapter_for, decoder_layers, load_model

Hooks = list[torch.utils.hooks.RemovableHandle]


@contextmanager
def hooked(handles: Hooks) -> Iterator[None]:
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def project_out(x: Tensor, unit: Tensor) -> Tensor:
    return x - (x @ unit)[..., None] * unit


def ablation_hooks(model: PreTrainedModel, unit: Tensor) -> Hooks:
    def pre(_module, args, kwargs):
        if args:
            return (project_out(args[0], unit), *args[1:]), kwargs
        kwargs["hidden_states"] = project_out(kwargs["hidden_states"], unit)
        return args, kwargs

    def post(_module, _args, output):
        if isinstance(output, tuple):
            return (project_out(output[0], unit), *output[1:])
        return project_out(output, unit)

    handles = []
    for block in decoder_layers(model):
        handles.append(block.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(block.self_attn.register_forward_hook(post))
        handles.append(block.mlp.register_forward_hook(post))
    return handles


def addition_hooks(model: PreTrainedModel, layer: int, vector: Tensor) -> Hooks:
    def pre(_module, args, kwargs):
        if args:
            return (args[0] + vector, *args[1:]), kwargs
        kwargs["hidden_states"] = kwargs["hidden_states"] + vector
        return args, kwargs

    return [decoder_layers(model)[layer].register_forward_pre_hook(pre, with_kwargs=True)]


def batches(rendered: list[list[int]], pad_id: int, size: int, left: bool = False) -> Iterator[tuple[Tensor, Tensor, list[int]]]:
    """(input_ids, attention_mask, last real position per row)."""
    for start in range(0, len(rendered), size):
        chunk = rendered[start : start + size]
        width = max(len(r) for r in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, r in enumerate(chunk):
            sl = slice(width - len(r), width) if left else slice(0, len(r))
            ids[j, sl] = torch.tensor(r)
            mask[j, sl] = 1
        yield ids, mask, [width - 1 if left else len(r) - 1 for r in chunk]


@torch.no_grad()
def category_sums(model, tokenizer, adapter, examples: list[Example], instructions: InstructionSet, batch_size: int) -> dict:
    """{category: (sum [position, layer, d_model] fp64, count)} of resid_pre at the post-instruction positions."""
    layers = decoder_layers(model)
    store: dict[int, Tensor] = {}
    handles = [b.register_forward_pre_hook(lambda m, a, k, i=i: store.__setitem__(i, (a[0] if a else k["hidden_states"]).detach()), with_kwargs=True) for i, b in enumerate(layers)]  # fmt: skip
    sums: dict[str, list] = {}
    rendered = [instructions.render(e, tokenizer, adapter) for e in examples]
    n_pos = len(rendered[0].positions)
    with hooked(handles):
        for start in range(0, len(examples), batch_size):
            chunk, rend = examples[start : start + batch_size], rendered[start : start + batch_size]
            ids, mask, _ = next(batches([r.input_ids for r in rend], tokenizer.pad_token_id, len(rend)))
            model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device), use_cache=False)
            for j, (e, r) in enumerate(zip(chunk, rend)):
                acts = torch.stack([store[i][j, r.positions].double().cpu() for i in range(len(layers))], 1)  # [pos, layer, d]
                total, count = sums.get(e.category, (torch.zeros(n_pos, len(layers), acts.shape[-1], dtype=torch.float64), 0))
                sums[e.category] = (total + acts, count + 1)
    return sums


def refusal_token_ids(tokenizer: PreTrainedTokenizerBase, tokens: list[str]) -> list[int]:
    ids = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in tokens]
    assert all(len(i) == 1 for i in ids), f"refusal tokens must be single tokens: {dict(zip(tokens, ids))}"
    return [i[0] for i in ids]


@torch.no_grad()
def last_logits(model, rendered: list[list[int]], pad_id: int, batch_size: int) -> Float[Tensor, "prompt vocab"]:
    out = []
    for ids, mask, last in batches(rendered, pad_id, batch_size):
        logits = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device), use_cache=False).logits
        out.append(logits[torch.arange(len(last)), last].float().cpu())
    return torch.cat(out)


def refusal_scores(logits: Tensor, refusal_ids: list[int], eps: float = 1e-8) -> Tensor:
    p = logits.softmax(-1)[:, refusal_ids].sum(-1)
    return torch.log(p + eps) - torch.log(1 - p + eps)


def kl_divergence(base: Tensor, other: Tensor) -> Tensor:
    lp, lq = base.log_softmax(-1), other.log_softmax(-1)
    return (lp.exp() * (lp - lq)).sum(-1)


def select_direction(
    model, cfg: DirectionRunConfig, diffs: Tensor, val: dict[str, list[list[int]]], refusal_ids: list[int], pad_id: int
) -> dict:
    """Scores every candidate (position, layer) on the val prompts; returns the grid and the choice."""
    s = cfg.selection
    target, other = cfg.direction.target, next(c for c in cfg.instructions.categories if c != cfg.direction.target)
    base_other = last_logits(model, val[other], pad_id, s.batch_size)
    n_pos, n_layers = diffs.shape[:2]
    max_layer = int(s.max_layer_fraction * n_layers)
    grid = []
    for p in range(n_pos):
        for layer in range(max_layer):
            if (
                diffs[p, layer].norm() < 1e-6
            ):  # e.g. layer 0: the input is the template's own embeddings, the same in every prompt
                continue
            vector = diffs[p, layer].to(model.device, model.dtype)
            unit = vector / vector.norm()
            with hooked(ablation_hooks(model, unit)):
                bypass = refusal_scores(last_logits(model, val[target], pad_id, s.batch_size), refusal_ids).mean()
                kl = kl_divergence(base_other, last_logits(model, val[other], pad_id, s.batch_size)).mean()
            with hooked(addition_hooks(model, layer, vector)):
                induce = refusal_scores(last_logits(model, val[other], pad_id, s.batch_size), refusal_ids).mean()
            grid.append(
                {"position": p - n_pos, "layer": layer, "bypass": float(bypass), "induce": float(induce), "kl": float(kl)}
            )
        print(f"[select] position {p - n_pos}: {max_layer} layers scored", flush=True)
    finite = [g for g in grid if all(map(math.isfinite, (g["bypass"], g["induce"], g["kl"])))]
    assert finite, "no candidate gave finite scores"
    kept = [g for g in finite if g["induce"] > 0 and g["kl"] < s.max_kl]
    choice = min(kept or finite, key=lambda g: g["bypass"])
    return {"grid": grid, "choice": choice, "filtered_out_all": not kept}


@torch.no_grad()
def generate(model, tokenizer, rendered: list[list[int]], cfg: DirectionRunConfig) -> list[str]:
    g = cfg.generation
    out = []
    for ids, mask, _ in batches(rendered, tokenizer.pad_token_id, g.batch_size, left=True):
        gen = model.generate(input_ids=ids.to(model.device), attention_mask=mask.to(model.device), max_new_tokens=g.max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.pad_token_id)  # fmt: skip
        out += [tokenizer.decode(row[ids.shape[1] :], skip_special_tokens=True) for row in gen]
    return out


def refusal_rate(replies: list[str], phrases: list[str]) -> float:
    return sum(any(p.lower() in r.lower() for p in phrases) for r in replies) / max(1, len(replies))


def generation_check(model, tokenizer, cfg: DirectionRunConfig, val: dict, vector: Tensor, layer: int) -> dict:
    target, other = cfg.direction.target, next(c for c in cfg.instructions.categories if c != cfg.direction.target)
    unit = vector / vector.norm()
    runs = {f"{target}/plain": (val[target], []), f"{target}/ablated": (val[target], ablation_hooks(model, unit)),
            f"{other}/plain": (val[other], []), f"{other}/added": (val[other], addition_hooks(model, layer, vector))}  # fmt: skip
    result = {}
    for name, (prompts, handles) in runs.items():
        with hooked(handles):
            replies = generate(model, tokenizer, prompts, cfg)
        result[name] = {"refusal_rate": refusal_rate(replies, cfg.generation.refusal_phrases), "replies": replies}
        print(f"[generate] {name}: refusal rate {result[name]['refusal_rate']:.2f}", flush=True)
    return result


def run_direction(cfg: DirectionRunConfig) -> dict:
    """GPU: results/<run_id>/{direction.json, direction.safetensors, replies.json}."""
    torch.set_grad_enabled(False)
    t0 = time.time()
    adapter: ModelAdapter = adapter_for(cfg.model)
    instructions = InstructionSet(cfg.instructions)
    examples = instructions.examples()
    model, tokenizer = load_model(cfg.model, cfg.device)
    train = [e for e in examples if e.split == "train"]
    sums = category_sums(model, tokenizer, adapter, train, instructions, cfg.selection.batch_size)
    categories = list(cfg.instructions.categories)
    means = torch.stack([sums[c][0] / sums[c][1] for c in categories])  # [category, position, layer, d]
    diffs = diff_from_means(means, categories, cfg)
    print(
        f"[means] {len(train)} train prompts, {diffs.shape[0]} positions x {diffs.shape[1]} layers | {time.time() - t0:.0f}s",
        flush=True,
    )

    val = {
        c: [instructions.render(e, tokenizer, adapter).input_ids for e in examples if e.split == "val" and e.category == c]
        for c in categories
    }
    refusal_ids = refusal_token_ids(tokenizer, cfg.selection.refusal_tokens)
    selection = select_direction(model, cfg, diffs, val, refusal_ids, tokenizer.pad_token_id)
    choice = selection["choice"]
    vector = diffs[choice["position"], choice["layer"]].to(model.device, model.dtype)
    print(
        f"[select] chose position {choice['position']}, layer {choice['layer']}: {choice} | {time.time() - t0:.0f}s", flush=True
    )
    check = generation_check(model, tokenizer, cfg, val, vector, choice["layer"])

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    save_file({"diffs": diffs.float().contiguous(), "means": means.float().contiguous(), "direction": vector.float().cpu().contiguous()},
              str(cfg.result_dir / "direction.safetensors"))  # fmt: skip
    summary = {
        "model": cfg.model, "categories": categories, "target": cfg.direction.target, "baseline": cfg.direction.baseline,
        "n_train": cfg.instructions.n_train, "n_val": cfg.instructions.n_val, "choice": choice,
        "filtered_out_all": selection["filtered_out_all"], "grid": selection["grid"],
        "refusal_rates": {k: v["refusal_rate"] for k, v in check.items()}, "seconds": round(time.time() - t0),
    }  # fmt: skip
    (cfg.result_dir / "direction.json").write_text(json.dumps(summary, indent=1))
    (cfg.result_dir / "replies.json").write_text(json.dumps({k: v["replies"] for k, v in check.items()}, indent=1))
    print(f"[saved] {cfg.result_dir} | {time.time() - t0:.0f}s", flush=True)
    return summary


def diff_from_means(means: Float[Tensor, "category ..."], categories: list[str], cfg: DirectionRunConfig) -> Tensor:
    """The difference of means from category means (the same rule as `vectors.diff_means`, without PCA)."""
    d = cfg.direction
    assert d.pca_reference is None, "PCA denoising needs the examples, not just their means"
    t = categories.index(d.target)
    rest = [i for i in range(len(categories)) if i != t]
    base = means.mean(0) if d.baseline == "all" else means[rest].mean(0)
    return means[t] - base
