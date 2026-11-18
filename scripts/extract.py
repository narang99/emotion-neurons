"""Stage 2a: one pass over every story, pooled sums of the layer-L sites.

    python scripts/extract.py --config configs/stage2_l25.yaml

Writes runs/<run_id>/sums.safetensors with, per site and range, an fp32 [n_stories, dim] sum
over that range's real tokens, plus `counts` [n_stories, 2] (tokens in early, late).
Stories are in load_stories order; runs/<run_id>/stories.json records it.
"""

import json
import subprocess
import sys
import time

import torch
from safetensors.torch import save_file
from tqdm import tqdm

from emotion_neurons.config import HOOKED_SITES, RANGES, load_run_config
from emotion_neurons.model import load_model, run_to_layer
from emotion_neurons.stories import load_stories


def pooled_sums(acts, mask, split):
    """Per story, the fp32 sum over real tokens before `split` and from `split` on."""
    pos = torch.arange(mask.shape[1], device=mask.device)
    ranges = {"early": mask & (pos < split), "late": mask & (pos >= split)}
    out = {}
    for site in HOOKED_SITES:
        a = acts[site].float()
        for name, m in ranges.items():
            out[f"{site}/{name}"] = (a * m[..., None]).sum(1)
    counts = torch.stack([ranges[r].sum(1) for r in RANGES], 1)
    return out, counts


def git_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def main():
    cfg = load_run_config(sys.argv[1:])
    t0 = time.time()
    stories = load_stories(cfg.data)
    model, tok = load_model(cfg.model, cfg.device)
    n_layers = len(model.model.layers)
    print(f"[load] {cfg.model}: {n_layers} layers, hooking layer {cfg.layer}, {len(stories)} stories, {time.time() - t0:.0f}s")

    lengths = [len(tok(s.text)["input_ids"]) for s in stories]
    assert max(lengths) <= cfg.extract.max_length, f"longest story is {max(lengths)} tokens"
    assert min(lengths) > cfg.data.split_token, f"shortest story is {min(lengths)} tokens"
    print(f"[tokens] min {min(lengths)}, median {sorted(lengths)[len(lengths) // 2]}, max {max(lengths)}")

    order = sorted(range(len(stories)), key=lambda i: -lengths[i])  # similar lengths share a batch
    sums: dict[str, torch.Tensor] = {}
    counts = torch.zeros(len(stories), 2, dtype=torch.int64)
    bs = cfg.extract.batch_size
    for start in tqdm(range(0, len(order), bs), desc="extract"):
        idx = order[start : start + bs]
        enc = tok([stories[i].text for i in idx], return_tensors="pt", padding=True).to(cfg.device)
        acts = run_to_layer(model, cfg.layer, enc)
        out, c = pooled_sums(acts, enc["attention_mask"].bool(), cfg.data.split_token)
        for k, v in out.items():
            if k not in sums:
                sums[k] = torch.zeros(len(stories), v.shape[1], dtype=torch.float32)
            sums[k][idx] = v.cpu()
        counts[idx] = c.cpu()
    print(f"[extract] done, {time.time() - t0:.0f}s")

    # Padding check: the story that got the most padding in its batch, run alone, must match its batched sums.
    pad = {j: max(lengths[k] for k in order[s : s + bs]) - lengths[j] for s in range(0, len(order), bs) for j in order[s : s + bs]}
    i = max(pad, key=pad.get)
    print(f"[padding check] {stories[i].id} had {pad[i]} pad tokens in its batch")
    enc = tok([stories[i].text], return_tensors="pt").to(cfg.device)
    out, c = pooled_sums(run_to_layer(model, cfg.layer, enc), enc["attention_mask"].bool(), cfg.data.split_token)
    assert torch.equal(c.cpu()[0], counts[i]), "token counts differ alone vs batched"
    for k, v in out.items():
        a, b = v.cpu()[0], sums[k][i]
        rel = ((a - b).norm() / b.norm()).item()
        print(f"[padding check] {stories[i].id} {k}: rel diff alone vs batched {rel:.2e}")
        assert rel < 2e-2, f"{k} differs alone vs batched ({rel:.2e}); padding leaks into the sums"

    assert counts.sum() == sum(lengths), "counted tokens != tokenized tokens"
    for k, v in sums.items():
        assert torch.isfinite(v).all(), f"non-finite values in {k}"

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    save_file({**sums, "counts": counts}, str(cfg.run_dir / "sums.safetensors"))
    meta = {
        "config": json.loads(cfg.model_dump_json()),
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "n_layers": n_layers,
        "dims": {k: v.shape[1] for k, v in sums.items()},
        "stories": [
            {"id": s.id, "emotion": s.emotion, "topic_id": s.topic_id, "story_idx": s.story_idx,
             "heldout": s.heldout, "n_tokens": n}
            for s, n in zip(stories, lengths)
        ],
    }  # fmt: skip
    (cfg.run_dir / "stories.json").write_text(json.dumps(meta, indent=1))
    print(f"[saved] {cfg.run_dir}/sums.safetensors, {len(sums)} arrays + counts")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
