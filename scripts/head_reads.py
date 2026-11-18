"""Which attention heads feed an MLP neuron, and a vocabulary read of the part of each head the neuron uses. GPU.

    python scripts/head_reads.py --config configs/head_reads.yaml

- Method in `emotion_neurons.circuits.heads`.
- For each neuron (the chosen ones and a few random ones of `random_layer`):
  1. its top `top_tokens` activations over the adult texts and the held-out stories;
  2. at those tokens, every head's exact push on its gate and up input, for all layers up to its own;
  3. heads ranked by first-order effect on the activation: `silu'(g) u * push_gate + silu(g) * push_up`;
  4. for the top heads, the logit lens of the used direction `u_h` (signed by the head's push), and of the head's
     whole write averaged over the tokens, for comparison;
  5. check: set the top heads' outputs to zero at those tokens, against random heads.
- Writes results/<run_id>/head_reads.json.
"""

import json
import random
import sys
import time

import torch

from emotion_neurons.circuits.heads import (
    activation_without_heads,
    head_count,
    head_pushes,
    head_reads,
    lane_read,
    read_heads,
    used_directions,
)
from emotion_neurons.config import load_run_config
from emotion_neurons.models import decoder_layers, load_model
from emotion_neurons.texts import load_labelled_texts, load_stories

torch.set_grad_enabled(False)
LANES = ("gate", "up")


def first_order_change(gate, up, d_gate, d_up):
    s = torch.sigmoid(gate)
    return s * (1 + gate * (1 - s)) * up * d_gate + torch.nn.functional.silu(gate) * d_up


def vocabulary(model, tokenizer, direction, k: int) -> list[str]:
    """Logit lens in the model's own precision: an fp32 copy of the unembedding needs 2.5 GB of GPU memory."""
    scaled = (model.model.norm.weight.float() * direction).to(model.lm_head.weight.dtype)
    logits = (model.lm_head.weight @ scaled).float()
    return [tokenizer.decode([i]) for i in logits.topk(k).indices.tolist()]


def top_tokens(model, sequences, layer: int, neuron: int, k: int) -> dict[int, torch.Tensor]:
    """Per sequence index, a mask of the tokens among the neuron's top-k activations (first 4 tokens excluded)."""
    found = []
    for i, ids in enumerate(sequences):
        acts = read_heads(model, ids, layer, [neuron]).activation[:, 0]
        acts[:4] = -float("inf")
        found += [(float(a), i, t) for t, a in enumerate(acts.tolist())]
    found.sort(reverse=True)
    masks: dict[int, torch.Tensor] = {}
    for _, i, t in found[:k]:
        masks.setdefault(i, torch.zeros(sequences[i].shape[1], dtype=torch.bool))[t] = True
    return masks


def neuron_heads(model, tokenizer, sequences, layer: int, neuron: int, cfg) -> dict:
    hr = cfg.head_reads
    n_heads, _ = head_count(model)
    masks = top_tokens(model, sequences, layer, neuron, hr.top_tokens)
    reads = {lane: lane_read(model, layer, neuron, lane) for lane in LANES}
    rows = {
        lane: (decoder_layers(model)[layer].mlp.gate_proj if lane == "gate" else decoder_layers(model)[layer].mlp.up_proj)
        .weight[neuron]
        .float()
        for lane in LANES
    }
    push = {lane: torch.zeros(layer + 1, n_heads) for lane in LANES}
    effect = torch.zeros(layer + 1, n_heads)
    write_sum = torch.zeros(layer + 1, n_heads, model.config.hidden_size)
    activation, count = 0.0, 0
    for i, tokens in masks.items():
        reading = read_heads(model, sequences[i], layer, [neuron])
        hits = tokens.to(reading.inverse_rms.device)
        lane_input = {lane: reading.mlp_input[hits] @ rows[lane] for lane in LANES}
        activation += float(reading.activation[hits, 0].sum())
        count += int(hits.sum())
        for k in range(layer + 1):
            p = {lane: head_pushes(model, reading.z[k], reading.inverse_rms, k, reads[lane])[hits] for lane in LANES}
            for lane in LANES:
                push[lane][k] += p[lane].sum(0).cpu()
            effect[k] += (
                first_order_change(lane_input["gate"][:, None], lane_input["up"][:, None], p["gate"], p["up"]).sum(0).cpu()
            )
            z = reading.z[k][hits].view(-1, n_heads, model.config.head_dim)
            W_O = decoder_layers(model)[k].self_attn.o_proj.weight.float().view(-1, n_heads, model.config.head_dim)
            write_sum[k] += torch.einsum("dhk,thk->hd", W_O, z).cpu()
    order = effect.abs().flatten().argsort(descending=True)
    top = [(int(i) // n_heads, int(i) % n_heads) for i in order[: hr.top_heads]]
    heads = []
    for k, h in top:
        lane = max(LANES, key=lambda x: abs(float(push[x][k, h])))
        u = used_directions(model, k, head_reads(model, k, reads[lane]))[h]
        sign = 1.0 if float(push[lane][k, h]) >= 0 else -1.0
        heads.append({
            "layer": k, "head": h, "effect": float(effect[k, h]), "push_gate": float(push["gate"][k, h]),
            "push_up": float(push["up"][k, h]), "lane": lane,
            "used_read": vocabulary(model, tokenizer, sign * u, hr.vocab_top),
            "whole_write_read": vocabulary(model, tokenizer, (write_sum[k, h] / count).to(u.device), hr.vocab_top),
        })  # fmt: skip
    checks = {}
    rng = random.Random(hr.seed + neuron)
    for size in hr.check_sizes:
        chosen = top[:size]
        random_heads = [(rng.randrange(layer + 1), rng.randrange(n_heads)) for _ in range(size)]
        checks[size] = {
            "top_heads_change": removed_change(model, sequences, masks, layer, neuron, chosen) - activation,
            "first_order_prediction": -float(sum(effect[k, h] for k, h in chosen)),
            "random_heads_change": removed_change(model, sequences, masks, layer, neuron, random_heads) - activation,
        }
    return {"layer": layer, "neuron": neuron, "hits": count, "activation_sum": activation, "heads": heads, "checks": checks,
            "attention_push_by_layer": {lane: push[lane].sum(1).tolist() for lane in LANES}}  # fmt: skip


def removed_change(model, sequences, masks, layer, neuron, heads) -> float:
    removed: dict[int, list[int]] = {}
    for k, h in heads:
        removed.setdefault(k, []).append(h)
    return sum(
        float(activation_without_heads(model, sequences[i], layer, neuron, removed, tokens).sum()) for i, tokens in masks.items()
    )


def main():
    cfg = load_run_config(sys.argv[1:])
    hr = cfg.head_reads
    t0 = time.time()
    model, tokenizer = load_model(cfg.model, cfg.device)
    texts = [t.text for t in load_labelled_texts(hr.texts_dir)] + [s.text for s in load_stories(cfg.data) if s.heldout]
    sequences = [tokenizer(t, return_tensors="pt")["input_ids"].to(cfg.device) for t in texts]
    rng = random.Random(hr.seed)
    neurons = [tuple(n) for n in hr.neurons] + [
        (hr.random_layer, rng.randrange(model.config.intermediate_size)) for _ in range(hr.random_count)
    ]
    print(f"[load] {len(sequences)} texts, neurons {neurons}, {time.time() - t0:.0f}s")
    out = []
    for layer, neuron in neurons:
        r = neuron_heads(model, tokenizer, sequences, layer, neuron, cfg)
        r["random"] = (layer, neuron) not in [tuple(n) for n in hr.neurons]
        out.append(r)
        c = r["checks"][hr.check_sizes[0]]
        print(f"[L{layer} n{neuron}] top head L{r['heads'][0]['layer']}h{r['heads'][0]['head']} reads {r['heads'][0]['used_read'][:5]} | "
              f"zero top {hr.check_sizes[0]}: {c['top_heads_change']:+.1f} (pred {c['first_order_prediction']:+.1f}), random {c['random_heads_change']:+.1f}", flush=True)  # fmt: skip
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "head_reads.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"[saved] {cfg.result_dir}/head_reads.json")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
