"""Pooled sums of every decoder layer's sites over every story, for following an emotion vector across depth. GPU.

    python scripts/extract_layers.py --config configs/layers_all.yaml

- Per layer l, summed over each story's late tokens (`data.split_token` on): `{l}/resid_mid`, `{l}/resid_post`,
  `{l}/mlp_act`; plus `embed`, the input to layer 0 (`emotion_neurons.models.late_sums_all_layers`).
- Checks:
  - late-token counts equal the tokenized lengths minus `split_token`;
  - the story with the most padding, run alone, gives the same sums.
- runs/<run_id>/: sums.safetensors (fp32 [story, dim] per key, plus `late_count` [story]), stories.json.
- Read back with `emotion_neurons.capture.LayerMeans`.
"""

import json
import subprocess
import sys
import time

import torch
from jaxtyping import Float
from safetensors.torch import save_file
from torch import Tensor
from tqdm import tqdm

from emotion_neurons.capture import split_masks
from emotion_neurons.config import load_run_config
from emotion_neurons.models import late_sums_all_layers, load_model
from emotion_neurons.texts import load_stories


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"


def batch_sums(model, tokenizer, texts: list[str], split_token: int, device: str):
    enc = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    late = split_masks(enc["attention_mask"], split_token)["late"]
    return late_sums_all_layers(model, enc["input_ids"], enc["attention_mask"], late), late.sum(1)


def main():
    cfg = load_run_config(sys.argv[1:])
    t0 = time.time()
    stories = load_stories(cfg.data)
    model, tokenizer = load_model(cfg.model, cfg.device)
    n_layers = len(model.model.layers)
    lengths = [len(tokenizer(s.text)["input_ids"]) for s in stories]
    assert max(lengths) <= cfg.extract.max_length and min(lengths) > cfg.data.split_token
    print(f"[load] {cfg.model}: {n_layers} layers, {len(stories)} stories, {time.time() - t0:.0f}s")

    order = sorted(range(len(stories)), key=lambda i: -lengths[i])
    sums: dict[str, Float[Tensor, "story dim"]] = {}
    late_count = torch.zeros(len(stories), dtype=torch.int64)
    bs = cfg.extract.batch_size
    for start in tqdm(range(0, len(order), bs), desc="extract"):
        idx = order[start : start + bs]
        batch, counts = batch_sums(model, tokenizer, [stories[i].text for i in idx], cfg.data.split_token, cfg.device)
        for key, value in batch.items():
            sums.setdefault(key, torch.zeros(len(stories), value.shape[1], dtype=torch.float32))[idx] = value.cpu()
        late_count[idx] = counts.cpu()
    print(f"[extract] {len(sums)} arrays, {time.time() - t0:.0f}s")

    assert torch.equal(late_count, torch.tensor([n - cfg.data.split_token for n in lengths])), "late-token counts differ"
    assert all(torch.isfinite(v).all() for v in sums.values()), "non-finite sums"
    pad = {
        j: max(lengths[k] for k in order[s : s + bs]) - lengths[j] for s in range(0, len(order), bs) for j in order[s : s + bs]
    }
    i = max(pad, key=pad.get)
    alone, _ = batch_sums(model, tokenizer, [stories[i].text], cfg.data.split_token, cfg.device)
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
        "stories": [s.model_dump(exclude={"text"}) | {"n_tokens": n} for s, n in zip(stories, lengths)],
    }
    (cfg.run_dir / "stories.json").write_text(json.dumps(meta, indent=1))
    print(f"[saved] {cfg.run_dir}/sums.safetensors, stories.json")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
