"""Stage 4: activations of chosen neurons (plus random controls) over a mixed text corpus. GPU.

    python scripts/harvest.py --config configs/stage3_l25.yaml

- Method in `emotion_neurons.harvest`.
- runs/<run_id>/: acts.npy [token, neuron] fp16, tokens.npy, pos.npy, window.npy, windows.json (corpus, doc id).
- results/<run_id>/: harvest.json (neurons, thresholds, examples), hits.safetensors (sampled p97+ hits per neuron).
"""

import json
import sys
import time

import numpy as np
import torch
from safetensors.torch import save_file

from emotion_neurons.config import load_run_config
from emotion_neurons.harvest import activation_pass, choose_neurons, collect_windows, neuron_bands, percentile_thresholds, snippet
from emotion_neurons.models import load_model


def main():
    cfg = load_run_config(sys.argv[1:])
    h = cfg.harvest
    t0 = time.time()
    model, tokenizer = load_model(cfg.model, cfg.device)
    W_down = model.model.layers[cfg.layer].mlp.down_proj.weight.float().cpu()
    choice = choose_neurons(cfg, W_down)
    neurons = choice.chosen + choice.controls
    print(f"[neurons] {len(choice.chosen)} from the set + {len(choice.controls)} controls = {len(neurons)}")

    windows = collect_windows(h, tokenizer)
    print(f"[corpus] total {int(windows.offsets[-1])} tokens in {len(windows.tokens)} windows, {time.time() - t0:.0f}s")
    rd = cfg.run_dir
    rd.mkdir(parents=True, exist_ok=True)
    arrays = activation_pass(model, cfg.layer, windows, neurons, h.batch_size, tokenizer.pad_token_id, rd / "acts.npy")
    np.save(rd / "tokens.npy", arrays.tokens)
    np.save(rd / "pos.npy", arrays.pos)
    np.save(rd / "window.npy", arrays.window)
    (rd / "windows.json").write_text(json.dumps(windows.corpus_doc))
    print(f"[harvest] done, {time.time() - t0:.0f}s")

    acts = np.load(rd / "acts.npy", mmap_mode="r")
    valid = arrays.pos >= h.skip_first
    valid_idx = np.nonzero(valid)[0]
    thresholds = percentile_thresholds(acts, valid, h, cfg.device)
    names = [c.name for c in h.corpora]
    corpus_of_window = np.array([names.index(c) for c, _ in windows.corpus_doc])
    rng = np.random.default_rng(h.seed)

    def describe(i: int) -> dict:
        return snippet(int(valid_idx[i]), windows, arrays, tokenizer).model_dump()

    examples, hits = {}, {}
    valid_t = torch.from_numpy(valid)
    for j, n in enumerate(neurons):
        values = torch.from_numpy(np.asarray(acts[:, j], dtype=np.float32))[valid_t].numpy()
        bands, hits[n], above = neuron_bands(values, thresholds[j], valid_idx, h, rng, describe)
        per_corpus = np.bincount(corpus_of_window[arrays.window[valid_idx[above]]], minlength=len(names))
        examples[n] = {"bands": bands, "p97_hits_by_corpus": dict(zip(names, map(int, per_corpus)))}

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "neurons": neurons, "controls": choice.controls, "groups": choice.groups, "percentiles": h.percentiles,
        "thresholds": {n: thresholds[j].tolist() for j, n in enumerate(neurons)}, "tokens": int(windows.offsets[-1]),
        "valid_tokens": int(valid.sum()), "corpora": [c.model_dump() for c in h.corpora], "examples": examples,
    }  # fmt: skip
    (cfg.result_dir / "harvest.json").write_text(json.dumps(out))
    save_file({str(n): torch.from_numpy(v.astype(np.int64)) for n, v in hits.items()}, str(cfg.result_dir / "hits.safetensors"))
    print(
        f"[saved] harvest.json ({len(neurons)} neurons), hits.safetensors; acts memmap {rd}/acts.npy {acts.shape[0]}x{acts.shape[1]}"
    )
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
