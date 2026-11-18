"""Stage 4b: modes of each chosen neuron's p97+ hits, by HDBSCAN on the neuron's input. GPU for the inputs.

    python scripts/modes.py --config configs/stage3_l25.yaml

- Neurons: those of `harvest.neurons_file` if given, else the top `modes.per_group` per (emotion, pole) of the
  Stage 3 ranking; plus the harvest's first `modes.n_controls` random controls.
- Method in `emotion_neurons.modes`.
- Writes results/<run_id>/modes.json, and modes/batch_XX.txt: exemplar snippets for Haiku to name.
"""

import json
import sys
import time

import numpy as np
from safetensors.torch import load_file

from emotion_neurons.config import RunConfig, load_run_config
from emotion_neurons.decomposition.dims_then_neurons import rank_neurons
from emotion_neurons.models import load_model
from emotion_neurons.modes import HarvestTokens, inputs_at, neuron_activation, neuron_modes
from emotion_neurons.vectors import EmotionVectors


def mode_groups(cfg: RunConfig, W_down, harvest: dict) -> dict[str, list[int]]:
    if cfg.harvest.neurons_file:
        groups = json.loads(cfg.harvest.neurons_file.read_text())["groups"]
    else:
        neuron_set = json.loads((cfg.result_dir / "neuron_set.json").read_text())
        vectors = EmotionVectors.load(cfg.results_dir / cfg.neuron_set.vectors_run / "vectors.safetensors")
        groups = rank_neurons(neuron_set, W_down, vectors.get("mlp_out"), vectors.get("mlp_act"), cfg.modes.per_group)
    groups["control"] = harvest["controls"][: cfg.modes.n_controls]
    harvested = set(harvest["neurons"])
    return {k: [n for n in v if n in harvested] for k, v in groups.items()}


def haiku_batches(chosen, result, harvest_tokens: HarvestTokens, acts, col, decode, per_batch: int) -> list[str]:
    texts = []
    for s in range(0, len(chosen), per_batch):
        lines = []
        for n in chosen[s : s + per_batch]:
            lines.append(f"##### neuron {n}")
            sections = [(c["id"], c["exemplars"]) for c in result[n]["clusters"]] + [("other", result[n]["other_exemplars"])]
            for key, exemplars in sections:
                if not exemplars:
                    continue
                lines.append(f"--- cluster {key}" if key != "other" else "--- unclustered (sample)")
                for t in exemplars:
                    before, token, after = harvest_tokens.snippet(t, decode)
                    lines.append(f"  [{float(acts[t, col[n]]):.2f}] {before}[[{token}]]{after}".replace("\n", " "))
        texts.append("\n".join(lines) + "\n")
    return texts


def main():
    cfg = load_run_config(sys.argv[1:])
    m = cfg.modes
    t0 = time.time()
    out_dir = cfg.result_dir
    harvest = json.loads((out_dir / "harvest.json").read_text())
    hits = {int(k): v.numpy() for k, v in load_file(str(out_dir / "hits.safetensors")).items()}
    harvest_tokens = HarvestTokens(cfg.run_dir)
    acts = np.load(cfg.run_dir / "acts.npy", mmap_mode="r")
    col = {n: i for i, n in enumerate(harvest["neurons"])}

    model, tokenizer = load_model(cfg.model, cfg.device)
    mlp = model.model.layers[cfg.layer].mlp
    groups = mode_groups(cfg, mlp.down_proj.weight.float().cpu(), harvest)
    chosen = sorted({n for g in groups.values() for n in g})
    print(
        f"[neurons] {len(chosen)} ({', '.join(f'{k}:{len(v)}' for k, v in groups.items() if k != 'control')}; controls {len(groups['control'])})"
    )

    needed = np.unique(np.concatenate([hits[n] for n in chosen])).tolist()
    X = inputs_at(model, cfg.layer, harvest_tokens, needed, cfg.harvest.batch_size, tokenizer.pad_token_id)
    t = needed[0]
    n = next(n for n in chosen if t in set(hits[n].tolist()))
    a = neuron_activation(mlp, n, X[t].float().to(cfg.device))
    print(
        f"[inputs] {len(X)} hit tokens, {time.time() - t0:.0f}s | check a_{n}: recomputed {a:.3f} vs harvested {float(acts[t, col[n]]):.3f}"
    )
    assert abs(a - float(acts[t, col[n]])) < 0.05 * max(1.0, abs(a)), "recomputed activation does not match the harvest"

    result = {}
    for n in chosen:
        ts = hits[n]
        result[n] = neuron_modes(n, ts, [X[t] for t in ts.tolist()], np.asarray(acts[ts, col[n]], dtype=np.float32), m)
        result[n]["groups"] = [g for g, v in groups.items() if n in v]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "modes.json").write_text(json.dumps({"groups": groups, "neurons": {str(k): v for k, v in result.items()}}))
    batch_dir = out_dir / "modes"
    batch_dir.mkdir(exist_ok=True)
    texts = haiku_batches(chosen, result, harvest_tokens, acts, col, tokenizer.decode, m.neurons_per_batch)
    for b, text in enumerate(texts):
        (batch_dir / f"batch_{b:02d}.txt").write_text(text)
    sizes = [len(r["clusters"]) for r in result.values()]
    print(f"[clusters] median {np.median(sizes):.0f} per neuron, {sum(s == 0 for s in sizes)} neurons with none; "
          f"median other share {np.median([r['other'] / r['n_hits'] for r in result.values()]):.2f}")  # fmt: skip
    print(f"[saved] modes.json, {len(texts)} batch files, ~{sum(len(t) for t in texts) // 4} tokens for Haiku")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
