"""Stage 2c: project every token of every held-out story onto every emotion vector. GPU.

    python scripts/project.py --config configs/stage2_l25.yaml

Reads results/<run_id>/vectors.safetensors. Writes to results/<run_id>/:
- projections.safetensors: `<site>/<baseline>/<variant>` [total_tokens, 12] fp16,
  token t of story j at offsets[j] + t; value = activation . unit vector
- projections_meta.json: story ids, labels, offsets, token strings
- logit_lens.json: top / bottom tokens of W_U (gamma * v) for the residual-stream vectors
"""

import json
import sys
import time

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from emotion_neurons.config import EMOTIONS, SITES, load_run_config
from emotion_neurons.model import load_model, run_to_layer
from emotion_neurons.stories import load_stories


def main():
    cfg = load_run_config(sys.argv[1:])
    t0 = time.time()
    stories = [s for s in load_stories(cfg.data) if s.heldout]
    vec = load_file(str(cfg.result_dir / "vectors.safetensors"))
    keys = [f"{site}/{v}" for site in SITES for v in cfg.project.vectors]
    units = {k: vec[k] / vec[k].norm(dim=1, keepdim=True) for k in keys}
    model, tok = load_model(cfg.model, cfg.device)
    units = {k: u.to(cfg.device) for k, u in units.items()}
    print(f"[load] {len(stories)} held-out stories, {len(keys)} vector sets, {time.time() - t0:.0f}s")

    proj = {k: [] for k in keys}
    tokens, lengths = [], []
    bs = cfg.project.batch_size
    for start in tqdm(range(0, len(stories), bs), desc="project"):
        batch = stories[start : start + bs]
        enc = tok([s.text for s in batch], return_tensors="pt", padding=True).to(cfg.device)
        acts = run_to_layer(model, cfg.layer, enc)
        acts = {k: v.float() for k, v in acts.items()}
        acts["mlp_out"] = acts["resid_post"] - acts["resid_mid"]
        n_real = enc["attention_mask"].sum(1).tolist()
        for k in keys:
            p = acts[k.split("/")[0]] @ units[k].T  # [b, seq, 12]
            for j, n in enumerate(n_real):
                proj[k].append(p[j, :n].cpu())
        for j, n in enumerate(n_real):
            ids = enc["input_ids"][j, :n].tolist()
            tokens.append([tok.decode([i]) for i in ids])
            lengths.append(n)

    out = {k: torch.cat(v).half().contiguous() for k, v in proj.items()}
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"{k} overflows fp16"
    offsets = [0]
    for n in lengths[:-1]:
        offsets.append(offsets[-1] + n)
    save_file(out, str(cfg.result_dir / "projections.safetensors"))
    pmeta = {
        "emotions": list(EMOTIONS),
        "keys": keys,
        "split_token": cfg.data.split_token,
        "stories": [
            {"id": s.id, "emotion": s.emotion, "topic_id": s.topic_id, "offset": o, "n_tokens": n, "tokens": t}
            for s, o, n, t in zip(stories, offsets, lengths, tokens)
        ],
    }
    (cfg.result_dir / "projections_meta.json").write_text(json.dumps(pmeta))
    print(f"[saved] projections for {len(stories)} stories, {sum(lengths)} tokens, {time.time() - t0:.0f}s")

    # Logit lens: what each residual-stream direction would push the next token towards.
    W_U = model.lm_head.weight.float()
    gamma = model.model.norm.weight.float()
    lens = {}
    k_top = cfg.project.logit_lens_k
    for site in ("resid_post", "resid_mid", "mlp_out"):
        for v in cfg.project.vectors:
            V = vec[f"{site}/{v}"].to(cfg.device).float()
            logits = (V / V.norm(dim=1, keepdim=True) * gamma) @ W_U.T  # [12, vocab]
            entry = {}
            for e, row in zip(EMOTIONS, logits):
                top = row.topk(k_top).indices.tolist()
                bot = (-row).topk(k_top).indices.tolist()
                entry[e] = {"top": [tok.decode([i]) for i in top], "bottom": [tok.decode([i]) for i in bot]}
            lens[f"{site}/{v}"] = entry
    (cfg.result_dir / "logit_lens.json").write_text(json.dumps(lens, indent=1, ensure_ascii=False))
    for e in EMOTIONS:
        print(f"[lens resid_post/{cfg.project.vectors[-1]}] {e}: {lens[f'resid_post/{cfg.project.vectors[-1]}'][e]['top'][:8]}")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
