"""Stage 2c: project every token of every held-out story onto every emotion vector. GPU.

    python scripts/project.py --config configs/stage2_l25.yaml

- Reads results/<run_id>/vectors.safetensors. Writes to results/<run_id>/:
  - projections.safetensors: `<site>/<baseline>/<variant>` [total_tokens, 12] fp16, token t of story j at
    offsets[j] + t; value = activation . unit vector;
  - projections_meta.json: story ids, labels, offsets, token strings;
  - logit_lens.json: top and bottom tokens of W_U (gamma * v) for the residual-stream vectors.
"""

import json
import sys
import time

import torch
from safetensors.torch import save_file
from tqdm import tqdm

from emotion_neurons.config import load_run_config
from emotion_neurons.emotions import EMOTIONS, SITES
from emotion_neurons.models import load_model, read_layer
from emotion_neurons.texts import load_stories
from emotion_neurons.vectors import EmotionVectors, unit


def logit_lens(model, tokenizer, vectors: EmotionVectors, keys: list[str], k_top: int, device: str) -> dict:
    """What each residual-stream direction would push the next token towards, final RMSNorm gain folded in."""
    W_U, gamma = model.lm_head.weight.float(), model.model.norm.weight.float()
    lens = {}
    for site in ("resid_post", "resid_mid", "mlp_out"):
        for v in keys:
            logits = (unit(vectors.tensors[f"{site}/{v}"].to(device).float()) * gamma) @ W_U.T
            lens[f"{site}/{v}"] = {
                e: {"top": [tokenizer.decode([i]) for i in row.topk(k_top).indices.tolist()],
                    "bottom": [tokenizer.decode([i]) for i in (-row).topk(k_top).indices.tolist()]}
                for e, row in zip(EMOTIONS, logits)
            }  # fmt: skip
    return lens


def main():
    cfg = load_run_config(sys.argv[1:])
    t0 = time.time()
    stories = [s for s in load_stories(cfg.data) if s.heldout]
    vectors = EmotionVectors.load(cfg.result_dir / "vectors.safetensors")
    keys = [f"{site}/{v}" for site in SITES for v in cfg.project.vectors]
    model, tokenizer = load_model(cfg.model, cfg.device)
    units = {k: unit(vectors.tensors[k]).to(cfg.device) for k in keys}
    print(f"[load] {len(stories)} held-out stories, {len(keys)} vector sets, {time.time() - t0:.0f}s")

    proj = {k: [] for k in keys}
    tokens, lengths = [], []
    bs = cfg.project.batch_size
    for start in tqdm(range(0, len(stories), bs), desc="project"):
        batch = stories[start : start + bs]
        enc = tokenizer([s.text for s in batch], return_tensors="pt", padding=True).to(cfg.device)
        acts = {k: v.float() for k, v in read_layer(model, cfg.layer, enc["input_ids"], enc["attention_mask"]).items()}
        acts["mlp_out"] = acts["resid_post"] - acts["resid_mid"]
        n_real = enc["attention_mask"].sum(1).tolist()
        for k in keys:
            p = acts[k.split("/")[0]] @ units[k].T
            proj[k] += [p[j, :n].cpu() for j, n in enumerate(n_real)]
        for j, n in enumerate(n_real):
            tokens.append([tokenizer.decode([i]) for i in enc["input_ids"][j, :n].tolist()])
            lengths.append(n)

    out = {k: torch.cat(v).half().contiguous() for k, v in proj.items()}
    assert all(torch.isfinite(v).all() for v in out.values()), "a projection overflows fp16"
    offsets = [0]
    for n in lengths[:-1]:
        offsets.append(offsets[-1] + n)
    save_file(out, str(cfg.result_dir / "projections.safetensors"))
    meta = {
        "emotions": list(EMOTIONS),
        "keys": keys,
        "split_token": cfg.data.split_token,
        "stories": [
            {"id": s.id, "emotion": s.emotion, "topic_id": s.topic_id, "offset": o, "n_tokens": n, "tokens": t}
            for s, o, n, t in zip(stories, offsets, lengths, tokens)
        ],
    }
    (cfg.result_dir / "projections_meta.json").write_text(json.dumps(meta))
    print(f"[saved] projections for {len(stories)} stories, {sum(lengths)} tokens, {time.time() - t0:.0f}s")

    lens = logit_lens(model, tokenizer, vectors, cfg.project.vectors, cfg.project.logit_lens_k, cfg.device)
    (cfg.result_dir / "logit_lens.json").write_text(json.dumps(lens, indent=1, ensure_ascii=False))
    last = cfg.project.vectors[-1]
    for e in EMOTIONS:
        print(f"[lens resid_post/{last}] {e}: {lens[f'resid_post/{last}'][e]['top'][:8]}")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
