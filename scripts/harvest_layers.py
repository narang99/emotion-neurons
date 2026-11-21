"""Harvest early-layer neurons in one pass: thresholds, top examples, top tokens, and an emotion-word share. GPU.

    python scripts/harvest_layers.py --config configs/harvest_layers.yaml

- Per layer in `layers`, the neurons are:
  - the first `per_emotion` neurons of each emotion's sparse set at that layer (`results/sparse_layers`, greedy order;
    an emotion whose set never matched the full MLP has none);
  - the `named` neurons of that layer (e.g. the chain's L8 n4826 and n7265);
  - `controls` random neurons.
- One forward pass over the harvest corpus reads all layers (stopped after the deepest).
- Cache: a layer whose `results/<run_id>/L<k>.json` exists is skipped, so a rerun adds only new layers.
- Per neuron: the percentile thresholds; the top 20 snippets; the counts of the tokens at its top 100 activations;
  `emotion_share`, the share of its top 50 tokens that are emotion words (a fixed word list; see EMOTION_STEMS).
- Writes results/<run_id>/L<k>.json, and the memmaps to runs/<run_id>/L<k>/acts.npy (synced to S3).
"""

import json
import random
import sys
import time
from collections import Counter

import numpy as np
import torch

from emotion_neurons.config import load_run_config
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.harvest import activation_pass_layers, collect_windows, percentile_thresholds, snippet
from emotion_neurons.models import load_model

EMOTION_STEMS = (
    "happ", "joy", "glad", "delight", "cheer", "inspir", "love", "loving", "affection", "proud", "pride", "calm", "peace",
    "seren", "relax", "desperat", "despair", "hopeless", "angr", "anger", "fury", "furious", "rage", "guilt", "ashamed",
    "shame", "sad", "sorrow", "grief", "griev", "afraid", "fear", "scare", "terrif", "fright", "nervous", "anxi", "worr",
    "surpris", "shock", "astonish", "amaz", "emotion", "embarrass", "excit", "upset", "lonel", "jealous", "envy",
)  # fmt: skip
EMOTION_CHARS = (
    "喜",
    "乐",
    "爱",
    "骄傲",
    "自豪",
    "平静",
    "绝望",
    "怒",
    "愤",
    "内疚",
    "愧",
    "悲",
    "伤心",
    "怕",
    "恐",
    "惧",
    "紧张",
    "焦虑",
    "惊",
    "羞",
    "耻",
    "情绪",
    "快乐",
    "幸福",
)


def is_emotion_word(token: str) -> bool:
    word = token.strip().lower()
    return bool(word) and (word.startswith(EMOTION_STEMS) or any(c in word for c in EMOTION_CHARS))


def choose(cfg, layer: int, rng: random.Random, d_mlp: int) -> tuple[list[int], dict[int, str]]:
    hl = cfg.harvest_layers
    sets = json.loads(hl.sparse_layers_file.read_text())["layers"][str(layer)]["emotions"]
    groups: dict[int, str] = {}
    for e in EMOTIONS:
        for n in (sets[e]["neurons"] or [])[: hl.per_emotion]:  # None: no set reached the full MLP's transfer
            groups.setdefault(int(n), e)
    for named_layer, n in hl.named:
        if named_layer == layer:
            groups[int(n)] = "named"
    pool = [n for n in range(d_mlp) if n not in groups]
    for n in rng.sample(pool, hl.controls):
        groups[n] = "control"
    return sorted(groups), groups


def main():
    cfg = load_run_config(sys.argv[1:])
    hl, h = cfg.harvest_layers, cfg.harvest
    t0 = time.time()
    todo = [k for k in hl.layers if not (cfg.result_dir / f"L{k}.json").exists()]
    if not todo:
        return print(f"[cache] all layers {hl.layers} done")
    print(f"[cache] to do: {todo}; done: {sorted(set(hl.layers) - set(todo))}")
    model, tokenizer = load_model(cfg.model, cfg.device)
    rng = random.Random(h.seed)
    chosen = {k: choose(cfg, k, rng, model.config.intermediate_size) for k in todo}
    windows = collect_windows(h, tokenizer)
    print(f"[corpus] {int(windows.offsets[-1])} tokens, {time.time() - t0:.0f}s", flush=True)
    paths = {k: cfg.run_dir / f"L{k}" / "acts.npy" for k in todo}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    arrays = activation_pass_layers(model, {k: chosen[k][0] for k in todo}, windows, h.batch_size, tokenizer.pad_token_id, paths)
    np.save(cfg.run_dir / "tokens.npy", arrays.tokens)
    print(f"[harvest] done, {time.time() - t0:.0f}s", flush=True)

    valid = arrays.pos >= h.skip_first
    valid_idx = np.nonzero(valid)[0]
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    for k in todo:
        neurons, groups = chosen[k]
        acts = np.load(paths[k])  # into memory: column reads of a memmap are slow
        thresholds = percentile_thresholds(acts, valid, h, cfg.device)
        per_neuron = {}
        for j, n in enumerate(neurons):
            values = np.asarray(acts[:, j], dtype=np.float32)[valid]
            top = np.argsort(-values)[: hl.top_tokens]
            words = [tokenizer.decode([int(arrays.tokens[valid_idx[i]])]) for i in top]
            per_neuron[n] = {
                "group": groups[n],
                "thresholds": thresholds[j].tolist(),
                "top_tokens": Counter(words).most_common(15),
                "emotion_share": sum(map(is_emotion_word, words[:50])) / 50,
                "examples": [{"act": float(values[i]), **snippet(int(valid_idx[i]), windows, arrays, tokenizer).model_dump()} for i in top[:20]],
            }  # fmt: skip

        def mean_share(keep) -> float:
            return float(np.mean([v["emotion_share"] for v in per_neuron.values() if keep(v["group"])]))

        share = {
            "emotion": mean_share(lambda g: g in EMOTIONS),
            "named": mean_share(lambda g: g == "named"),
            "control": mean_share(lambda g: g == "control"),
        }
        out = {
            "layer": k,
            "percentiles": h.percentiles,
            "tokens": int(windows.offsets[-1]),
            "emotion_share_mean": share,
            "neurons": per_neuron,
        }
        (cfg.result_dir / f"L{k}.json").write_text(json.dumps(out, ensure_ascii=False))
        print(
            f"[L{k}] {len(neurons)} neurons, mean emotion share: set neurons {share['emotion']:.2f}, named {share['named']:.2f}, controls {share['control']:.2f}",
            flush=True,
        )
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
