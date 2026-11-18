"""The local circuit of one neuron on the tokens where it fires: its writers, its readers, and checks. GPU.

    python scripts/neuron_circuit.py --config configs/neuron_circuit_3068.yaml

- Hit tokens: every token past the first 4 where the neuron is at or above its harvest p97, over the adult texts and
  the held-out stories.
- Writers (`emotion_neurons.circuits.neuron.split_input`): its gate and up inputs split exactly into the embeddings,
  each layer's attention, and each earlier neuron.
  - Writers are ranked by their first-order effect on the neuron's activation through `silu(gate) * up`.
  - Check: hold the top writers at their mean at the hit tokens and compare the activation change with the
    first-order prediction, against random writers of the same number.
- Readers: every neuron of the later layers, ranked by the first-order change of its activation caused by this
  neuron's write; then each reader's value is read through the unembedding.
  - Check: hold this neuron at its mean and compare the top readers' activation change with the prediction.
- Causal readout: with the neuron held at its mean at the hit tokens, the change in next-token log-probabilities.
- Writes results/<run_id>/neuron_circuit.json.
"""

import json
import sys
import time

import torch
from safetensors.torch import load_file

from emotion_neurons.circuits.neuron import (
    activation_with_held,
    first_order_change,
    lane_inputs,
    read_stream,
    split_input,
)
from emotion_neurons.config import load_run_config
from emotion_neurons.models import decoder_layers, load_model
from emotion_neurons.texts import load_labelled_texts, load_stories

torch.set_grad_enabled(False)


def texts_to_scan(cfg) -> list[tuple[str, str]]:
    adult = [(t.id, t.text) for t in load_labelled_texts(cfg.neuron_circuit.texts_dir)]
    held = [(s.id, s.text) for s in load_stories(cfg.data) if s.heldout]
    return adult + held


def value_tokens(model, tokenizer, layer: int, neuron: int, k: int = 8) -> list[str]:
    write = decoder_layers(model)[layer].mlp.down_proj.weight[:, neuron].float()
    logits = model.lm_head.weight.float() @ (model.model.norm.weight.float() * write)
    return [tokenizer.decode([i]) for i in logits.topk(k).indices.tolist()]


def snippet(tokenizer, ids, t: int) -> str:
    return (
        tokenizer.decode(ids[max(0, t - 12) : t])
        + "[["
        + tokenizer.decode(ids[t : t + 1])
        + "]]"
        + tokenizer.decode(ids[t + 1 : t + 4])
    ).replace("\n", " ")


def writer_effects(model, stream, layer, neuron, tokens, means):
    """Per earlier layer, each neuron's first-order effect on the target's activation, summed over the hit tokens.

    - Measured against the writer's mean activation, since the check holds writers at their mean.
    """
    gate_in, up_in = (lane_inputs(model, stream, layer, tokens)[lane][:, neuron] for lane in ("gate", "up"))
    blocks = decoder_layers(model)
    scale = stream.inverse_rms[layer][tokens]
    effects = []
    for k in range(layer):
        down = blocks[k].mlp.down_proj.weight.float()
        V = {lane: down.T @ (row * blocks[layer].post_attention_layernorm.weight.float())
             for lane, row in (("gate", blocks[layer].mlp.gate_proj.weight[neuron].float()), ("up", blocks[layer].mlp.up_proj.weight[neuron].float()))}  # fmt: skip
        shift = stream.activation[k][tokens] - means[k].to(stream.activation[k].device)[None, :]
        share = {lane: shift * scale[:, None] * V[lane][None, :] for lane in V}
        effects.append(first_order_change(gate_in[:, None], up_in[:, None], share["gate"], share["up"]).sum(0).cpu())
    return torch.stack(effects)


def reader_effects(model, stream, layer, neuron, reader_layer, tokens, mean: float):
    """Each later neuron's first-order activation change from this neuron's write, against its mean, summed over hits."""
    block = decoder_layers(model)[reader_layer]
    write = decoder_layers(model)[layer].mlp.down_proj.weight[:, neuron].float()
    gain = block.post_attention_layernorm.weight.float()
    amount = (stream.activation[layer][tokens, neuron] - float(mean)) * stream.inverse_rms[reader_layer][tokens]
    z = lane_inputs(model, stream, reader_layer, tokens)
    share = {lane: amount[:, None] * ((proj.weight.float() * gain) @ write)[None, :]
             for lane, proj in (("gate", block.mlp.gate_proj), ("up", block.mlp.up_proj))}  # fmt: skip
    return first_order_change(z["gate"], z["up"], share["gate"], share["up"]).sum(0).cpu()


def run_with(model, ids, reader_layers, hold):
    """Next-token log-probabilities and the reader layers' activations; `hold` = (layer, neuron, value, tokens)."""
    store, handles = {}, []
    for k in reader_layers:
        handles.append(
            decoder_layers(model)[k].mlp.down_proj.register_forward_pre_hook(
                lambda _m, a, k=k: store.__setitem__(k, a[0][0].float())
            )
        )
    if hold is not None:
        layer, neuron, value, tokens = hold

        def set_to_mean(_m, a):
            x = a[0].clone()
            x[0, tokens.to(x.device), neuron] = value
            return (x,)

        handles.insert(0, decoder_layers(model)[layer].mlp.down_proj.register_forward_pre_hook(set_to_mean))
    try:
        logp = model(input_ids=ids).logits[0].float().log_softmax(-1)
    finally:
        for h in handles:
            h.remove()
    return logp, store


def main():
    cfg = load_run_config(sys.argv[1:])
    nc = cfg.neuron_circuit
    t0 = time.time()
    model, tokenizer = load_model(cfg.model, cfg.device)
    threshold = json.loads((cfg.results_dir / nc.harvest_run / "harvest.json").read_text())["thresholds"][str(nc.neuron)][0]
    means = load_file(str(cfg.circuits.vectors_file))["mlp_act_mean"]
    L, n = nc.layer, nc.neuron
    last = len(decoder_layers(model)) - 1

    totals = {"writers": torch.zeros(L, means.shape[1]), "attention": torch.zeros(L + 1), "embed": 0.0, "mlp": 0.0,
              "actual": {"gate": 0.0, "up": 0.0}, "rebuilt": {"gate": 0.0, "up": 0.0},
              "readers": torch.zeros(last - L, means.shape[1])}  # fmt: skip
    hits, sequences = [], []
    for text_id, text in texts_to_scan(cfg):
        ids = tokenizer(text, return_tensors="pt")["input_ids"].to(cfg.device)
        stream = read_stream(model, ids, last)
        acts = stream.activation[L][:, n]
        tokens = (acts >= threshold) & (torch.arange(len(acts), device=acts.device) >= 4)
        if not tokens.any():
            continue
        sequences.append((text_id, ids, tokens.cpu()))
        id_list = ids[0].tolist()
        hits += [
            {"text": text_id, "act": float(acts[t]), "context": snippet(tokenizer, id_list, t)}
            for t in tokens.nonzero().flatten().tolist()
        ]
        for lane in ("gate", "up"):
            split = split_input(model, stream, L, n, lane, tokens)
            totals["actual"][lane] += split.actual
            totals["rebuilt"][lane] += split.rebuilt()
            if lane == "gate":
                totals["attention"] += torch.tensor(split.attention)
                totals["embed"] += split.embed
                totals["mlp"] += split.mlp_total()
        totals["writers"] += writer_effects(model, stream, L, n, tokens, means)
        for i, reader_layer in enumerate(range(L + 1, last + 1)):
            totals["readers"][i] += reader_effects(model, stream, L, n, reader_layer, tokens, means[L, n])
    print(f"[hits] {len(hits)} tokens in {len(sequences)} texts, {time.time() - t0:.0f}s")

    # Writers: top by |first-order effect| on the neuron's activation.
    flat = totals["writers"].abs().flatten().topk(nc.top_writers).indices
    width = totals["writers"].shape[1]
    writers = [(int(i) // width, int(i) % width) for i in flat]
    writer_rows = [
        {"layer": k, "neuron": j, "effect": float(totals["writers"][k, j]), "value": value_tokens(model, tokenizer, k, j)}
        for k, j in writers
    ]

    def held_activation(held_pairs):
        held = {}
        for k, j in held_pairs:
            held.setdefault(k, []).append(j)
        held = {k: torch.tensor(v) for k, v in held.items()}
        total = 0.0
        for _, ids, tokens in sequences:
            total += float(activation_with_held(model, ids, L, n, means, held, tokens).sum())
        return total

    clean = sum(h["act"] for h in hits)
    checks = {}
    for k in nc.check_sizes:
        top = writers[:k]
        predicted = -float(sum(totals["writers"][a, b] for a, b in top))
        generator = torch.Generator().manual_seed(k)
        random_pairs = [
            (int(torch.randint(0, L, (1,), generator=generator)), int(torch.randint(0, width, (1,), generator=generator)))
            for _ in range(k)
        ]
        checks[k] = {"top_writers_change": held_activation(top) - clean, "first_order_prediction": predicted,
                     "random_writers_change": held_activation(random_pairs) - clean}  # fmt: skip
        print(
            f"[writers] hold top {k}: activation change {checks[k]['top_writers_change']:+.1f} (predicted {predicted:+.1f}); random {checks[k]['random_writers_change']:+.1f} | clean total {clean:.1f}"
        )

    # Readers: top by |first-order change| per hit token.
    flat = totals["readers"].abs().flatten().topk(nc.top_readers).indices
    readers = [(L + 1 + int(i) // width, int(i) % width) for i in flat]
    reader_rows = [
        {
            "layer": k,
            "neuron": j,
            "change_per_hit": float(totals["readers"][k - L - 1, j]),
            "value": value_tokens(model, tokenizer, k, j),
        }
        for k, j in readers
    ]

    # Causal readout: next-token log-probabilities and reader activations, clean vs the neuron held at its mean.
    reader_layers = sorted({k for k, _ in readers})
    shift = torch.zeros(len(tokenizer))
    reader_actual = torch.zeros(len(readers))
    for _, ids, tokens in sequences:
        clean_logp, clean_acts = run_with(model, ids, reader_layers, None)
        held_logp, held_acts = run_with(model, ids, reader_layers, (L, n, float(means[L, n]), tokens))
        rows = tokens.to(clean_logp.device)
        shift += (held_logp[rows] - clean_logp[rows]).sum(0).cpu()
        for i, (k, j) in enumerate(readers):
            reader_actual[i] += float((held_acts[k][rows, j] - clean_acts[k][rows, j]).sum())
    for i, row in enumerate(reader_rows):
        row["actual_change_when_held_per_hit"] = float(reader_actual[i]) / len(hits)
    down = shift.topk(nc.top_tokens, largest=False)
    up = shift.topk(nc.top_tokens)
    readout = {"most_lowered": [[tokenizer.decode([i]), float(v) / len(hits)] for v, i in zip(down.values.tolist(), down.indices.tolist())],
               "most_raised": [[tokenizer.decode([i]), float(v) / len(hits)] for v, i in zip(up.values.tolist(), up.indices.tolist())]}  # fmt: skip

    out = {
        "layer": L, "neuron": n, "threshold": threshold, "n_hits": len(hits), "n_texts": len(sequences),
        "hits": sorted(hits, key=lambda h: -h["act"])[:40], "input_split": {
            "actual": totals["actual"], "rebuilt": totals["rebuilt"], "embed": totals["embed"],
            "attention_by_layer": totals["attention"].tolist(), "mlp_neurons": totals["mlp"]},
        "writers": writer_rows, "writer_checks": checks, "readers": reader_rows, "readout": readout,
        "value": value_tokens(model, tokenizer, L, n, 12),
    }  # fmt: skip
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "neuron_circuit.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"[saved] {cfg.result_dir}/neuron_circuit.json")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
