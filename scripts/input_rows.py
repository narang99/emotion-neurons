"""What does a neuron listen for, read from its weights alone? Geva's lens on the MLP input rows. CPU.

    python scripts/input_rows.py --config configs/input_rows.yaml

- A neuron of layer l reads the stream x as `a = silu(W_gate[n] . (g * x/rms)) * (W_up[n] . (g * x/rms))`.
- Two lenses put a token's direction in place of x and compute `a` for every token of the vocabulary:
  - `embed`: x = the token's input embedding `W_E[tok]`. The neuron's response to the token alone.
  - `unembed`: x = `g_final * W_U[tok]`, the stream direction that predicts the token.
- Check against data: for neurons with a harvest (`results/harvest_layers`), the share of their 10 most frequent
  top-activation tokens that are among the lens' top 200 tokens.
  - Baseline: the same, with the lens of another neuron of the same layer (shuffled).
- Also read the chain and circuit neurons (`named`), both ways: the tokens with the most positive `a`, and with
  the most negative `a` (a neuron such as L2 n8663 fires negatively on its words).
- Writes results/<run_id>/input_rows.json.
"""

import json
import random
import sys

import torch
from transformers import AutoTokenizer

from emotion_neurons.config import load_run_config
from emotion_neurons.models.weights import load_cached_bf16, load_cached_tensor

torch.set_grad_enabled(False)


def lens_activations(gate, up, gain, lens: tuple, chunk: int = 8192) -> torch.Tensor:
    """`a` [vocab, neuron] for every row of X (times `scale`, if any) taken as the stream, after the RMS norm."""
    X, scale = lens
    out = []
    for c in range(0, X.shape[0], chunk):
        x = X[c : c + chunk].float()
        if scale is not None:
            x = x * scale
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        x = x * gain
        out.append(torch.nn.functional.silu(x @ gate.T) * (x @ up.T))
    return torch.cat(out)


def overlap(found: list[str], lens_top: set[str]) -> float:
    return sum(t in lens_top for t in found) / max(1, len(found))


def main():
    cfg = load_run_config(sys.argv[1:])
    ir = cfg.input_rows
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    n_vocab = len(tokenizer)
    g_final = load_cached_tensor(cfg.model, "model.norm.weight")
    lenses = {"embed": (load_cached_bf16(cfg.model, "model.embed_tokens.weight")[:n_vocab], None),
              "unembed": (load_cached_bf16(cfg.model, "lm_head.weight")[:n_vocab], g_final)}  # fmt: skip
    decoded = [tokenizer.decode([i]) for i in range(n_vocab)]
    rng = random.Random(0)
    out = {"layers": {}, "named": {}}

    harvested = {int(p.stem[1:]): json.loads(p.read_text())["neurons"] for p in sorted(ir.harvest_dir.glob("L*.json"))}
    wanted: dict[int, set[int]] = {k: {int(n) for n in v} for k, v in harvested.items()}
    for layer, n in ir.named:
        wanted.setdefault(layer, set()).add(n)

    for layer in sorted(wanted):
        neurons = sorted(wanted[layer])
        gate = load_cached_tensor(cfg.model, f"model.layers.{layer}.mlp.gate_proj.weight")[neurons]
        up = load_cached_tensor(cfg.model, f"model.layers.{layer}.mlp.up_proj.weight")[neurons]
        gain = load_cached_tensor(cfg.model, f"model.layers.{layer}.post_attention_layernorm.weight")
        tops, bottoms = {}, {}
        for name, X in lenses.items():
            a = lens_activations(gate, up, gain, X)
            tops[name] = {n: a[:, j].topk(ir.top_k).indices.tolist() for j, n in enumerate(neurons)}
            bottoms[name] = {n: (-a[:, j]).topk(ir.show).indices.tolist() for j, n in enumerate(neurons)}
        if layer in harvested:
            h = harvested[layer]
            rows = {}
            for name in lenses:
                real, shuffled = [], []
                ids = [int(n) for n in h]
                for n in ids:
                    found = [t for t, _ in h[str(n)]["top_tokens"][:10]]
                    other = rng.choice([m for m in ids if m != n])
                    real.append(overlap(found, {decoded[i] for i in tops[name][n]}))
                    shuffled.append(overlap(found, {decoded[i] for i in tops[name][other]}))
                rows[name] = {"overlap": sum(real) / len(real), "shuffled": sum(shuffled) / len(shuffled),
                              "neurons_with_any_overlap": sum(r > 0 for r in real) / len(real)}  # fmt: skip
            out["layers"][layer] = rows
            print(f"[L{layer}] {len(h)} harvested neurons | " + " | ".join(
                f"{k}: overlap {v['overlap']:.3f} (shuffled {v['shuffled']:.3f}), any {v['neurons_with_any_overlap']:.2f}" for k, v in rows.items()), flush=True)  # fmt: skip
        for named_layer, n in ir.named:
            if named_layer == layer:
                out["named"][f"L{layer} n{n}"] = {name: [decoded[i] for i in tops[name][n][: ir.show]] for name in lenses}
                out["named"][f"L{layer} n{n}"] |= {f"{name}_negative": [decoded[i] for i in bottoms[name][n]] for name in lenses}
                print(
                    f"  L{layer} n{n}: embed {out['named'][f'L{layer} n{n}']['embed'][:10]} | unembed {out['named'][f'L{layer} n{n}']['unembed'][:10]}",
                    flush=True,
                )
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "input_rows.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[saved] {cfg.result_dir}/input_rows.json")


if __name__ == "__main__":
    main()
