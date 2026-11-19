"""Find the path from a token to a target neuron by patching: which layers, then which neurons, carry the effect. GPU.

    python scripts/chain_scan.py --config configs/chain.yaml

- Hits: the target's top tokens (as in `chain.py`). At each hit t, the token at t is replaced by `replacement`.
  - The stream at t is `embed(t) + sum_k attn_k(t) + sum_k mlp_k(t)`, and every path from token t to the target
    at t goes through these terms, so patching them at t alone is a complete scan.
- Layer scan: patch one layer's MLP output (or attention output) at t back to its clean value, with the token
  still replaced. `recovery = (patched - swapped) / (clean - swapped)` of the target's activation.
- Neuron scan, in the `scan_top_layers` MLP layers with the largest mean recovery:
  - candidates: the `candidates` neurons with the largest |change| of activation under the swap, at that hit;
  - patch each one alone back to its clean value; record its recovery;
  - baseline: `random_middles` random neurons of the same layer, patched the same way.
- Then the best neurons are tested as a set, and held at their mean on the clean text (the link test of `chain.py`).
- Writes results/<run_id>/chain_scan.json.
"""

import json
import random
import re
import sys
import time
from collections.abc import Callable

import torch
from safetensors.torch import load_file

from emotion_neurons.config import load_run_config
from emotion_neurons.models import decoder_layers, load_model
from emotion_neurons.models.qwen3 import StopForward, hooks, raise_stop

torch.set_grad_enabled(False)
sys.path.insert(0, "scripts")
from chain import Node, hits_of  # noqa: E402

from emotion_neurons.texts import load_labelled_texts, load_stories  # noqa: E402


class Patches:
    """Hooks that read or overwrite, at token t, layer outputs (`mlp`, `attn`) and neuron activations (`act`)."""

    def __init__(self, model, t: int):
        self.model, self.t = model, t

    def run(self, ids, target: Node, set_out: dict, set_act: dict, keep_all: bool = False) -> tuple[float, dict]:
        """Target activation at t, with {(kind, layer): vector} outputs and {(layer, neuron): value} activations set at t."""
        blocks = decoder_layers(self.model)
        t = self.t
        seen: dict = {"mlp": {}, "attn": {}, "act": {}}

        def on_out(kind: str, k: int) -> Callable:
            def hook(_m, _a, out):
                h = out[0] if isinstance(out, tuple) else out
                if (kind, k) in set_out:
                    h = h.clone()
                    h[0, t] = set_out[(kind, k)].to(h.dtype)
                if keep_all:
                    seen[kind][k] = h[0, t].clone()
                return (h, *out[1:]) if isinstance(out, tuple) else h

            return hook

        def on_act(k: int) -> Callable:
            def hook(_m, args):
                a = args[0]
                mine = [(n, v) for (layer, n), v in set_act.items() if layer == k]
                if mine:
                    a = a.clone()
                    for n, v in mine:
                        a[0, t, n] = v
                if keep_all or k == target.layer:
                    seen["act"][k] = a[0, t].float().clone()
                return (a,)

            return hook

        handles = []
        for k, b in enumerate(blocks[: target.layer + 1]):
            handles += [b.mlp.down_proj.register_forward_pre_hook(on_act(k))]
            if k < target.layer:
                handles += [b.mlp.register_forward_hook(on_out("mlp", k))]
            handles += [b.self_attn.register_forward_hook(on_out("attn", k))]
        handles.append(blocks[target.layer].register_forward_hook(raise_stop))
        with hooks(handles):
            try:
                self.model.model(input_ids=ids, use_cache=False)
            except StopForward:
                pass
        return float(seen["act"][target.layer][target.neuron]), seen


def main():
    cfg = load_run_config(sys.argv[1:])
    ch = cfg.chain
    t0 = time.time()
    model, tokenizer = load_model(cfg.model, cfg.device)
    act_means = load_file(str(ch.layer_vectors_file))["mlp_act_mean"].float()
    target = Node(*ch.target)
    replacement = tokenizer(ch.replacement, add_special_tokens=False)["input_ids"][0]
    texts = [t.text for t in load_labelled_texts(ch.texts_dir)] + [s.text for s in load_stories(cfg.data) if s.heldout]
    sequences = [tokenizer(t, return_tensors="pt")["input_ids"].to(cfg.device) for t in texts]
    threshold = json.loads((cfg.results_dir / ch.harvest_run / "harvest.json").read_text())["thresholds"][str(target.neuron)][
        ch.band
    ]
    hits = hits_of(model, sequences, target, threshold, 10_000)
    if ch.exclude_pattern:
        # Keep only hits whose own token does not match, e.g. context hits of a word neuron.
        hits = [
            h
            for h in hits
            if not re.match(ch.exclude_pattern, tokenizer.decode(sequences[h[0]][0, h[1] : h[1] + 1]).strip().lower())
        ]
    hits = hits[: ch.max_hits]
    L = target.layer
    rng = random.Random(ch.seed)
    print(f"[hits] {len(hits)}, {time.time() - t0:.0f}s", flush=True)
    # Hits where replacing the token lowers the target by less than `min_drop` are skipped: their recovery is noise.

    # Layer scan.
    rows = []
    for i, t, _ in hits:
        ids = sequences[i]
        swapped = ids.clone()
        swapped[0, t] = replacement
        p = Patches(model, t)
        clean, clean_seen = p.run(ids, target, {}, {}, keep_all=True)
        swap, swap_seen = p.run(swapped, target, {}, {}, keep_all=True)
        drop = clean - swap
        if drop < ch.min_drop:
            continue
        rec = {kind: [(p.run(swapped, target, {(kind, k): clean_seen[kind][k]}, {})[0] - swap) / drop for k in range(L if kind == "mlp" else L + 1)]
               for kind in ("mlp", "attn")}  # fmt: skip
        act_change = {k: (clean_seen["act"][k] - swap_seen["act"][k]).cpu() for k in range(L)}
        rows.append({"text": i, "t": t, "token": tokenizer.decode(ids[0, t : t + 1]), "clean": clean, "swapped": swap,
                     "recovery": rec, "_clean_acts": {k: clean_seen["act"][k].cpu() for k in range(L)}, "_act_change": act_change})  # fmt: skip
    mean_rec = {
        kind: [sum(r["recovery"][kind][k] for r in rows) / len(rows) for k in range(len(rows[0]["recovery"][kind]))]
        for kind in ("mlp", "attn")
    }
    top_layers = sorted(range(L), key=lambda k: -mean_rec["mlp"][k])[: ch.scan_top_layers]
    print(f"[layer scan] mean target drop {sum(r['clean'] - r['swapped'] for r in rows) / len(rows):.2f}; "
          f"MLP recovery top: {[(k, round(mean_rec['mlp'][k], 2)) for k in top_layers]}; "
          f"attention top: {sorted(((k, round(v, 2)) for k, v in enumerate(mean_rec['attn'])), key=lambda x: -x[1])[:4]}, {time.time() - t0:.0f}s", flush=True)  # fmt: skip

    # Neuron scan in the top MLP layers.
    neuron_rec: dict[tuple[int, int], list[float]] = {}
    random_rec: dict[int, list[float]] = {k: [] for k in top_layers}
    for r in rows:
        i, t = r["text"], r["t"]
        ids = sequences[i]
        swapped = ids.clone()
        swapped[0, t] = replacement
        p = Patches(model, t)
        drop = r["clean"] - r["swapped"]
        for k in top_layers:
            change = r["_act_change"][k]
            for n in change.abs().argsort(descending=True)[: ch.candidates].tolist():
                v = p.run(swapped, target, {}, {(k, n): float(r["_clean_acts"][k][n])})[0]
                neuron_rec.setdefault((k, n), []).append((v - r["swapped"]) / drop)
            for n in rng.sample(range(act_means.shape[1]), ch.random_middles):
                v = p.run(swapped, target, {}, {(k, n): float(r["_clean_acts"][k][n])})[0]
                random_rec[k].append((v - r["swapped"]) / drop)
    # Average over all hits (a neuron that was not a candidate at a hit counts as 0 there).
    best = sorted(neuron_rec, key=lambda kn: -sum(neuron_rec[kn]) / len(rows))[: ch.report_neurons]
    rand_mean = {k: sum(v) / max(1, len(v)) for k, v in random_rec.items()}
    print(f"[neuron scan] random neuron recovery per layer { ({k: round(v, 4) for k, v in rand_mean.items()}) }", flush=True)

    # The best neurons as a set: patch all of them back after the swap; and hold them at their mean on clean text.
    set_rec, hold_change = [], []
    for r in rows:
        i, t = r["text"], r["t"]
        ids = sequences[i]
        swapped = ids.clone()
        swapped[0, t] = replacement
        p = Patches(model, t)
        patch = {(k, n): float(r["_clean_acts"][k][n]) for k, n in best}
        set_rec.append((p.run(swapped, target, {}, patch)[0] - r["swapped"]) / (r["clean"] - r["swapped"]))
        hold = {(k, n): float(act_means[k, n]) for k, n in best}
        hold_change.append(p.run(ids, target, {}, hold)[0] - r["clean"])
    rand_set = [(k, n) for k, n in ((rng.choice(top_layers), rng.randrange(act_means.shape[1])) for _ in best)]
    rand_hold = []
    for r in rows:
        p = Patches(model, r["t"])
        rand_hold.append(
            p.run(sequences[r["text"]], target, {}, {(k, n): float(act_means[k, n]) for k, n in rand_set})[0] - r["clean"]
        )

    def value(k: int, n: int) -> list[str]:
        write = decoder_layers(model)[k].mlp.down_proj.weight[:, n].float()
        logits = model.lm_head.weight @ (model.model.norm.weight.float() * write).to(model.lm_head.weight.dtype)
        return [tokenizer.decode([j]) for j in logits.float().topk(8).indices.tolist()]

    out = {
        "target": ch.target, "replacement": ch.replacement, "n_hits_scanned": len(rows), "hits": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        "mean_recovery": mean_rec, "top_layers": top_layers, "random_neuron_recovery": rand_mean,
        "best_neurons": [{"layer": k, "neuron": n, "recovery": sum(neuron_rec[(k, n)]) / len(rows), "candidate_hits": len(neuron_rec[(k, n)]),
                          "value": value(k, n)} for k, n in best],
        "best_set": {"recovery": sum(set_rec) / len(set_rec), "hold_change": sum(hold_change) / len(hold_change),
                     "random_set_hold_change": sum(rand_hold) / len(rand_hold), "target_mean": sum(r["clean"] for r in rows) / len(rows)},
    }  # fmt: skip
    b = out["best_set"]
    print(f"[best set] {len(best)} neurons: patched recovery {b['recovery']:.2f}; held at mean: target {b['hold_change']:+.2f} "
          f"(random set {b['random_set_hold_change']:+.2f}) of {b['target_mean']:.2f}", flush=True)  # fmt: skip
    for x in out["best_neurons"][:10]:
        print(
            f"  L{x['layer']} n{x['neuron']} recovery {x['recovery']:.3f} ({x['candidate_hits']} hits) value {x['value'][:5]}",
            flush=True,
        )
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "chain_scan.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"[saved] {cfg.result_dir}/chain_scan.json")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
