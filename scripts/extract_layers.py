"""Pooled sums of every decoder layer's sites over every story, for following an emotion vector across depth. GPU.

    python scripts/extract_layers.py --config configs/layers_all.yaml

- Sites per layer l, summed over each story's late tokens (from `data.split_token` on):
  - `{l}/resid_mid`: the stream after attention, the input to post_attention_layernorm.
  - `{l}/resid_post`: the layer's output.
  - `{l}/mlp_act`: the MLP neurons, silu(gate) * up, the input to down_proj.
- `embed` is the input to layer 0, so attention's write at layer l is `{l}/resid_mid - {l-1}/resid_post`.
- The sums are taken inside the hooks.
  - Keeping every layer's full activations for a batch would need about 8 GB of GPU memory.
- Checks:
  - the late-token counts equal the tokenized lengths minus `split_token`;
  - the story with the most padding, run alone, gives the same sums.

runs/<run_id>/: sums.safetensors (fp32, [n_stories, dim] per key, plus `late_count` [n_stories]), stories.json.
"""

import json
import subprocess
import sys
import time

import torch
from einops import einsum
from jaxtyping import Bool, Float
from safetensors.torch import save_file
from torch import Tensor
from tqdm import tqdm

from emotion_neurons.config import load_run_config
from emotion_neurons.model import load_model
from emotion_neurons.stories import load_stories

LayerSums = dict[str, Float[Tensor, "batch dim"]]


def late_sum(x: Float[Tensor, "batch seq dim"], late: Bool[Tensor, "batch seq"]) -> Float[Tensor, "batch dim"]:
    return einsum(x.float(), late.float(), "batch seq dim, batch seq -> batch dim")


def pooled_sums_all_layers(model, input_ids, attention_mask, late: Bool[Tensor, "batch seq"]) -> LayerSums:
    """One full forward pass with a hook per site per layer, each adding its late-token sum to `sums`."""
    sums: LayerSums = {}

    def keep(name):
        return lambda _module, args: sums.__setitem__(name, late_sum(args[0], late))

    def keep_output(name):
        return lambda _module, _args, out: sums.__setitem__(name, late_sum(out, late))

    handles = [model.model.layers[0].register_forward_pre_hook(keep("embed"))]
    for i, layer in enumerate(model.model.layers):
        handles += [
            layer.post_attention_layernorm.register_forward_pre_hook(keep(f"{i}/resid_mid")),
            layer.mlp.down_proj.register_forward_pre_hook(keep(f"{i}/mlp_act")),
            layer.register_forward_hook(keep_output(f"{i}/resid_post")),
        ]
    try:
        with torch.no_grad():
            model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return sums


def late_mask(attention_mask: Tensor, split_token: int) -> Bool[Tensor, "batch seq"]:
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    return attention_mask.bool() & (positions >= split_token)


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"


def main():
    cfg = load_run_config(sys.argv[1:])
    t0 = time.time()
    stories = load_stories(cfg.data)
    model, tok = load_model(cfg.model, cfg.device)
    n_layers = len(model.model.layers)
    lengths = [len(tok(s.text)["input_ids"]) for s in stories]
    assert max(lengths) <= cfg.extract.max_length and min(lengths) > cfg.data.split_token
    print(f"[load] {cfg.model}: {n_layers} layers, {len(stories)} stories, {time.time() - t0:.0f}s")

    order = sorted(range(len(stories)), key=lambda i: -lengths[i])
    sums: dict[str, Float[Tensor, "story dim"]] = {}
    late_count = torch.zeros(len(stories), dtype=torch.int64)
    bs = cfg.extract.batch_size
    for start in tqdm(range(0, len(order), bs), desc="extract"):
        idx = order[start : start + bs]
        enc = tok([stories[i].text for i in idx], return_tensors="pt", padding=True).to(cfg.device)
        late = late_mask(enc["attention_mask"], cfg.data.split_token)
        batch = pooled_sums_all_layers(model, enc["input_ids"], enc["attention_mask"], late)
        for key, value in batch.items():
            if key not in sums:
                sums[key] = torch.zeros(len(stories), value.shape[1], dtype=torch.float32)
            sums[key][idx] = value.cpu()
        late_count[idx] = late.sum(1).cpu()
    print(f"[extract] {len(sums)} arrays, {time.time() - t0:.0f}s")

    expected = torch.tensor([n - cfg.data.split_token for n in lengths])
    assert torch.equal(late_count, expected), "late-token counts differ from the tokenized lengths"
    assert all(torch.isfinite(v).all() for v in sums.values()), "non-finite sums"

    pad = {j: max(lengths[k] for k in order[s : s + bs]) - lengths[j] for s in range(0, len(order), bs) for j in order[s : s + bs]}
    i = max(pad, key=pad.get)
    enc = tok([stories[i].text], return_tensors="pt").to(cfg.device)
    alone = pooled_sums_all_layers(model, enc["input_ids"], enc["attention_mask"], late_mask(enc["attention_mask"], cfg.data.split_token))
    worst = max(((alone[k].cpu()[0] - sums[k][i]).norm() / sums[k][i].norm()).item() for k in alone)
    print(f"[padding check] {stories[i].id} ({pad[i]} pad tokens): worst rel diff alone vs batched {worst:.2e}")
    assert worst < 2e-2, "padding leaks into the sums"

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    save_file({**sums, "late_count": late_count}, str(cfg.run_dir / "sums.safetensors"))
    meta = {
        "config": json.loads(cfg.model_dump_json()),
        "git_commit": git_commit(),
        "n_layers": n_layers,
        "keys": "embed; {layer}/resid_mid, {layer}/resid_post, {layer}/mlp_act: late-token sums [n_stories, dim]",
        "stories": [
            {"id": s.id, "emotion": s.emotion, "topic_id": s.topic_id, "story_idx": s.story_idx,
             "heldout": s.heldout, "n_tokens": n}
            for s, n in zip(stories, lengths)
        ],
    }  # fmt: skip
    (cfg.run_dir / "stories.json").write_text(json.dumps(meta, indent=1))
    print(f"[saved] {cfg.run_dir}/sums.safetensors, stories.json")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
