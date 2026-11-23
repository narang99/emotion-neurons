"""Stage 2a: one pass over every story, pooled sums of the layer-L sites. GPU.

    python scripts/extract.py --config experiments/00-vector-extraction/stage2_l25.yaml

- runs/<run_id>/sums.safetensors: per site and range (early, late), an fp32 [story, dim] sum over that range's real
  tokens, plus `counts` [story, 2].
- runs/<run_id>/stories.json records the story order (`emotion_neurons.texts.load_stories`).
- Checks:
  - the story with the most padding in its batch, run alone, gives the same sums;
  - the counted tokens equal the tokenized tokens.
"""

import json
import subprocess
import sys
import time

import torch
from safetensors.torch import save_file
from tqdm import tqdm

from emotion_neurons.capture import pooled_sums
from emotion_neurons.config import VectorsRunConfig, load_run_config
from emotion_neurons.models import load_model, read_layer
from emotion_neurons.texts import load_stories


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"


def story_sums(model, tokenizer, layer: int, texts: list[str], split_token: int, device: str):
    enc = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    acts = read_layer(model, layer, enc["input_ids"], enc["attention_mask"])
    return pooled_sums(acts, enc["attention_mask"], split_token)


def pad_tokens_per_story(lengths: list[int], order: list[int], batch_size: int) -> dict[int, int]:
    """Pad tokens each story gets: its batch's longest story minus its own length. Batches follow `order`."""
    pad = {}
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        longest = max(lengths[i] for i in batch)
        pad.update({i: longest - lengths[i] for i in batch})
    return pad


def padding_check(model, tokenizer, cfg, stories, lengths, order, sums, counts) -> None:
    """The most padded story, run alone, must give the batched sums: padding must not leak into them."""
    pad = pad_tokens_per_story(lengths, order, cfg.extract.batch_size)
    i = max(pad, key=pad.get)
    print(f"[padding check] {stories[i].id} had {pad[i]} pad tokens in its batch")
    alone, c = story_sums(model, tokenizer, cfg.layer, [stories[i].text], cfg.data.split_token, cfg.device)
    assert torch.equal(c.cpu()[0], counts[i]), "token counts differ alone vs batched"
    for site, v in alone.items():
        rel = ((v.cpu()[0] - sums[site][i]).norm() / sums[site][i].norm()).item()
        print(f"[padding check] {stories[i].id} {site}: rel diff alone vs batched {rel:.2e}")
        assert rel < 2e-2, f"{site} differs alone vs batched ({rel:.2e}); padding leaks into the sums"


def token_lengths(tokenizer, stories, cfg) -> list[int]:
    lengths = [len(tokenizer(s.text)["input_ids"]) for s in stories]
    assert max(lengths) <= cfg.extract.max_length, f"longest story is {max(lengths)} tokens"
    assert min(lengths) > cfg.data.split_token, f"shortest story is {min(lengths)} tokens"
    print(f"[tokens] min {min(lengths)}, median {sorted(lengths)[len(lengths) // 2]}, max {max(lengths)}")
    return lengths


def extract_all(model, tokenizer, cfg, stories, order):
    """Pooled sums for every story, batched in `order`. Returns `sums` per site and `counts` [story, 2]."""
    sums: dict[str, torch.Tensor] = {}
    counts = torch.zeros(len(stories), 2, dtype=torch.int64)
    bs = cfg.extract.batch_size
    for start in tqdm(range(0, len(order), bs), desc="extract"):
        idx = order[start : start + bs]
        texts = [stories[i].text for i in idx]
        out, c = story_sums(model, tokenizer, cfg.layer, texts, cfg.data.split_token, cfg.device)
        for site, v in out.items():
            sums.setdefault(site, torch.zeros(len(stories), v.shape[1], dtype=torch.float32))[idx] = v.cpu()
        counts[idx] = c.cpu()
    return sums, counts


def save_run(cfg, stories, lengths, sums, counts, n_layers: int) -> None:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    save_file({**sums, "counts": counts}, str(cfg.run_dir / "sums.safetensors"))
    meta = {
        "config": json.loads(cfg.model_dump_json()),
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "n_layers": n_layers,
        "dims": {k: v.shape[1] for k, v in sums.items()},
        "stories": [s.model_dump(exclude={"text"}) | {"n_tokens": n} for s, n in zip(stories, lengths)],
    }
    (cfg.run_dir / "stories.json").write_text(json.dumps(meta, indent=1))
    print(f"[saved] {cfg.run_dir}/sums.safetensors, {len(sums)} arrays + counts")


def main():
    cfg = load_run_config(VectorsRunConfig, sys.argv[1:])
    t0 = time.time()
    stories = load_stories(cfg.data)
    model, tokenizer = load_model(cfg.model, cfg.device)
    n_layers = len(model.model.layers)
    print(f"[load] {cfg.model}: {n_layers} layers, hooking layer {cfg.layer}, {len(stories)} stories, {time.time() - t0:.0f}s")

    lengths = token_lengths(tokenizer, stories, cfg)
    order = sorted(range(len(stories)), key=lambda i: -lengths[i])  # similar lengths share a batch
    sums, counts = extract_all(model, tokenizer, cfg, stories, order)
    print(f"[extract] done, {time.time() - t0:.0f}s")

    padding_check(model, tokenizer, cfg, stories, lengths, order, sums, counts)
    assert counts.sum() == sum(lengths), "counted tokens != tokenized tokens"
    assert all(torch.isfinite(v).all() for v in sums.values()), "non-finite sums"

    save_run(cfg, stories, lengths, sums, counts, n_layers)
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
