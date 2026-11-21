"""A chain from input tokens to a logit: token -> middle neuron -> target neuron -> next-token log-prob. GPU.

    python scripts/chain.py --config configs/chain.yaml

- Hits: tokens where the target neuron (L25 n3068, "pride") is at or above its harvest threshold, over the adult
  texts and the held-out stories. At most `max_hits`, strongest first.
- For each middle neuron (e.g. L8 n4826, "proud/自豪") and each hit token t:
  1. **Where the middle neuron's input comes from** (`circuits.sources.split_by_source`, exact):
     its own token's embedding, each earlier MLP, and each head from each source token.
     The top source token s* is the one with the largest summed effect through attention, or t itself when
     the embedding plus MLPs at t is larger.
  2. **Link middle -> target**: hold the middle neuron at its mean at t; the target's change.
     Baseline: the same for `random_middles` random neurons of the middle layer.
  3. **Link target -> logit**: hold the target at its mean at t; the change in log p(`logit_token`) at t.
  4. **Token -> everything**: replace the token at s* with `replacement`; the change of the middle, the target and
     the log-prob. Baseline: replace `random_tokens` random other tokens among the 20 before t.
  5. **Mediation**: replace s*, but patch the middle neuron (then the target) back to its clean value at t.
     The recovered share of the target's (the log-prob's) change is what flows through that node.
- Writes results/<run_id>/chain.json.
"""

import json
import random
import sys
import time
from collections.abc import Callable

import torch
from safetensors.torch import load_file

from emotion_neurons.circuits.sources import SourceSplit, split_by_source
from emotion_neurons.config import load_run_config
from emotion_neurons.models import decoder_layers, load_model
from emotion_neurons.models.qwen3 import StopForward, hooks, raise_stop
from emotion_neurons.texts import load_labelled_texts, load_stories

torch.set_grad_enabled(False)


class Node:
    """An MLP neuron whose activation at one token is read, and optionally set."""

    def __init__(self, layer: int, neuron: int):
        self.layer, self.neuron = layer, neuron

    def __repr__(self) -> str:
        return f"L{self.layer} n{self.neuron}"


def forward(model, ids, t: int, read: list[Node], set_to: dict, logit_token: int) -> tuple[list[float], float]:
    """Activations of `read` nodes at t and log p(logit_token) at t, with `set_to` {(layer, neuron): value} applied at t."""
    blocks = decoder_layers(model)
    seen: dict = {}
    by_layer: dict[int, list] = {}
    for (layer, neuron), value in set_to.items():
        by_layer.setdefault(layer, []).append((neuron, value))
    layers = sorted({n.layer for n in read} | set(by_layer))

    def hook_for(layer: int) -> Callable:
        def hook(_m, args):
            a = args[0]
            for neuron, value in by_layer.get(layer, []):
                a = a.clone() if a is args[0] else a
                a[0, t, neuron] = value
            for n in read:
                if n.layer == layer:
                    seen[(n.layer, n.neuron)] = float(a[0, t, n.neuron])
            return (a,)

        return hook

    with hooks([blocks[k].mlp.down_proj.register_forward_pre_hook(hook_for(k)) for k in layers]):
        logits = model(input_ids=ids, use_cache=False).logits[0, t].float()
    return [seen[(n.layer, n.neuron)] for n in read], float(logits.log_softmax(-1)[logit_token])


def hits_of(model, sequences, target: Node, threshold: float, max_hits: int) -> list[tuple[int, int, float]]:
    found = []
    for i, ids in enumerate(sequences):
        store = {}
        block = decoder_layers(model)[target.layer]
        keep = block.mlp.down_proj.register_forward_pre_hook(
            lambda _m, a: store.__setitem__("a", a[0][0, :, target.neuron].float())
        )
        with hooks([keep, block.register_forward_hook(raise_stop)]):
            try:
                model.model(input_ids=ids, use_cache=False)
            except StopForward:
                pass
        acts = store["a"]
        found += [(i, t, float(acts[t])) for t in range(4, len(acts)) if acts[t] >= threshold]
    return sorted(found, key=lambda x: -x[2])[:max_hits]


def combined_split(model, ids, middle: Node, t: int) -> SourceSplit:
    """Both lanes' splits, weighted by their first-order effect on the activation `silu(gate) * up`."""
    gate, up = (split_by_source(model, ids, middle.layer, middle.neuron, lane, t) for lane in ("gate", "up"))
    g, u = torch.tensor(gate.lane_input), torch.tensor(up.lane_input)
    w_gate = float(torch.sigmoid(g) * (1 + g * (1 - torch.sigmoid(g))) * u)
    w_up = float(torch.nn.functional.silu(g))
    return SourceSplit(
        lane_input=w_gate * gate.lane_input + w_up * up.lane_input,
        embed=w_gate * gate.embed + w_up * up.embed,
        mlp=w_gate * gate.mlp + w_up * up.mlp,
        head_source=w_gate * gate.head_source + w_up * up.head_source,
    )


def top_heads(split: SourceSplit, k: int) -> list[tuple[int, int, float]]:
    per_head = split.head_source.sum(-1)
    order = per_head.flatten().abs().argsort(descending=True)[:k]
    return [(int(j) // per_head.shape[1], int(j) % per_head.shape[1], float(per_head.flatten()[j])) for j in order]


def top_source(split, t: int) -> tuple[int, dict]:
    """The token whose input moves the middle neuron most, by magnitude (a neuron can fire negatively)."""
    by_source = split.by_source()
    own = float(by_source[t]) + split.embed + float(split.mlp.sum())
    others = by_source.abs()
    others[t] = -1.0
    j = int(others.argmax())
    s = j if float(others[j]) > abs(own) else t
    return s, {"own_token": own, "best_other": float(by_source[j]), "best_other_at": j - t}


def main():
    cfg = load_run_config(sys.argv[1:])
    ch = cfg.chain
    t0 = time.time()
    model, tokenizer = load_model(cfg.model, cfg.device)
    model.set_attn_implementation("eager")
    act_means = load_file(str(ch.layer_vectors_file))["mlp_act_mean"].float()
    target = Node(*ch.target)
    logit_token = tokenizer(ch.logit_token, add_special_tokens=False)["input_ids"][0]
    replacement = tokenizer(ch.replacement, add_special_tokens=False)["input_ids"][0]
    texts = [t.text for t in load_labelled_texts(ch.texts_dir)] + [s.text for s in load_stories(cfg.data) if s.heldout]
    sequences = [tokenizer(t, return_tensors="pt")["input_ids"].to(cfg.device) for t in texts]
    threshold = json.loads((cfg.results_dir / ch.harvest_run / "harvest.json").read_text())["thresholds"][str(target.neuron)][
        ch.band
    ]
    hits = hits_of(model, sequences, target, threshold, ch.max_hits)
    print(f"[hits] {len(hits)} hits of {target} at >= {threshold:.2f}, {time.time() - t0:.0f}s", flush=True)
    rng = random.Random(ch.seed)
    out = {
        "target": ch.target,
        "logit_token": ch.logit_token,
        "replacement": ch.replacement,
        "threshold": threshold,
        "middles": {},
    }

    for middle_pair in ch.middles:
        middle = Node(*middle_pair)
        mean_mid, mean_tgt = float(act_means[middle.layer, middle.neuron]), float(act_means[target.layer, target.neuron])
        randoms = [Node(middle.layer, n) for n in rng.sample(range(act_means.shape[1]), ch.random_middles)]
        rows = []
        for i, t, _ in hits:
            ids = sequences[i]
            (mid, tgt), lp = forward(model, ids, t, [middle, target], {}, logit_token)
            split = combined_split(model, ids, middle, t)
            s, source_info = top_source(split, t)
            row = {"text": i, "t": t, "token": tokenizer.decode(ids[0, t : t + 1]), "middle": mid, "target": tgt, "logp": lp,
                   "context": tokenizer.decode(ids[0, max(0, t - 10) : t + 1]).replace("\n", " "),
                   "source": s - t, "source_token": tokenizer.decode(ids[0, s : s + 1]), "source_info": source_info,
                   "split": {"embed": split.embed, "mlp": float(split.mlp.sum()), "mlp_by_layer": split.mlp.tolist(), "attention": float(split.head_source.sum()),
                             "attention_from_own_token": float(split.head_source[..., t].sum()),
                             "top_heads": top_heads(split, 3)}}  # fmt: skip
            # Link middle -> target, and its random baseline.
            (tgt_held,), _ = forward(model, ids, t, [target], {(middle.layer, middle.neuron): mean_mid}, logit_token)
            row["hold_middle_target_change"] = tgt_held - tgt
            row["hold_random_target_change"] = [
                forward(model, ids, t, [target], {(r.layer, r.neuron): float(act_means[r.layer, r.neuron])}, logit_token)[0][0]
                - tgt
                for r in randoms
            ]
            # Link target -> logit.
            _, lp_held = forward(model, ids, t, [], {(target.layer, target.neuron): mean_tgt}, logit_token)
            row["hold_target_logp_change"] = lp_held - lp
            # Token -> everything, with mediation; and random tokens.
            swapped = ids.clone()
            swapped[0, s] = replacement
            (mid_s, tgt_s), lp_s = forward(model, swapped, t, [middle, target], {}, logit_token)
            (tgt_med,), lp_med_mid = forward(model, swapped, t, [target], {(middle.layer, middle.neuron): mid}, logit_token)
            _, lp_med_tgt = forward(model, swapped, t, [], {(target.layer, target.neuron): tgt}, logit_token)
            row["swap"] = {"middle": mid_s - mid, "target": tgt_s - tgt, "logp": lp_s - lp,
                           "target_with_middle_patched": tgt_med - tgt, "logp_with_middle_patched": lp_med_mid - lp,
                           "logp_with_target_patched": lp_med_tgt - lp}  # fmt: skip
            pool = [p for p in range(max(1, t - 20), t + 1) if p != s]
            row["swap_random"] = []
            for p in rng.sample(pool, min(ch.random_tokens, len(pool))):
                other = ids.clone()
                other[0, p] = replacement
                (m_r, g_r), lp_r = forward(model, other, t, [middle, target], {}, logit_token)
                row["swap_random"].append({"offset": p - t, "middle": m_r - mid, "target": g_r - tgt, "logp": lp_r - lp})
            rows.append(row)
        out["middles"][str(middle)] = {"mean": mean_mid, "random_middles": [str(r) for r in randoms], "hits": rows}
        print(summary_line(middle, rows), f"{time.time() - t0:.0f}s", flush=True)

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "chain.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"[saved] {cfg.result_dir}/chain.json")
    print(f"[total] {time.time() - t0:.0f}s")


def summary_line(middle: Node, rows: list[dict]) -> str:
    def mean(xs):
        return sum(xs) / max(1, len(xs))

    rand_hold = mean([mean([abs(x) for x in r["hold_random_target_change"]]) for r in rows])
    rand_swap = mean([mean([x["target"] for x in r["swap_random"]]) for r in rows])
    own = mean([r["source"] == 0 for r in rows])
    return (f"[{middle}] hits {len(rows)} | hold middle: target {mean([r['hold_middle_target_change'] for r in rows]):+.2f} "
            f"(random |change| {rand_hold:.2f}) | hold target: logp {mean([r['hold_target_logp_change'] for r in rows]):+.2f} | "
            f"swap source (own token {own:.2f}): middle {mean([r['swap']['middle'] for r in rows]):+.2f} target {mean([r['swap']['target'] for r in rows]):+.2f} "
            f"logp {mean([r['swap']['logp'] for r in rows]):+.2f}, random token: target {rand_swap:+.2f} | "
            f"middle patched: target {mean([r['swap']['target_with_middle_patched'] for r in rows]):+.2f}")  # fmt: skip


if __name__ == "__main__":
    main()
