"""How each emotion vector is built across the 36 layers: attention vs MLP, direction, decodability. CPU.

    python scripts/journey.py --config configs/layers_all.yaml

- Reads runs/<run_id>/{sums.safetensors, stories.json} from `extract_layers.py`.
- Method in `emotion_neurons.journey`.
- Writes results/<run_id>/:
  - journey.json: per-layer readings, build-up of the layer-25 and last-layer vectors, cosine to both;
  - journey.md: the same as tables (means over the 12 emotions);
  - charts/*.svg.
"""

import json
import sys
import time

from emotion_neurons.capture import LayerMeans
from emotion_neurons.config import load_run_config
from emotion_neurons.journey import BuildUp, LayerReading, build_up, cos_to, read_layers
from emotion_neurons.svg import line_chart
from emotion_neurons.texts import StorySplit, load_story_records

REFERENCE_LAYERS = (25,)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def cumulative(per_layer: list[list[float]]) -> list[float]:
    out, total = [], 0.0
    for shares in per_layer:
        total += mean(shares)
        out.append(total)
    return out


def charts(readings: list[LayerReading], builds: dict[int, BuildUp], cos: dict[int, dict[int, list[float]]]) -> dict[str, str]:
    out = {
        "added_per_layer": line_chart(
            "Added at each layer, along that layer's vector (share of |v|)",
            {"attention": [mean(r.attention_along) for r in readings], "MLP": [mean(r.mlp_along) for r in readings]},
        ),
        "decodability": line_chart(
            "Held-out 12-way accuracy, and |v| relative to the last layer",
            {
                "accuracy": [r.heldout_accuracy for r in readings],
                "|v| / |v_last|": [mean(r.norm_denoised) / mean(readings[-1].norm_denoised) for r in readings],
            },
        ),
        "direction": line_chart(
            "Cosine with the vector at a reference layer, and with the previous layer",
            {f"layer {ref}": [mean(c) for c in by_layer.values()] for ref, by_layer in cos.items()}
            | {"previous": [mean(r.cos_previous) for r in readings]},
        ),
    }
    for ref, b in builds.items():
        out[f"build_up_{ref}"] = line_chart(
            f"Build-up of the layer-{ref} vector: cumulative share written up to each layer",
            {"attention": cumulative(b.attention), "MLP": cumulative(b.mlp)},
        )
    return out


def markdown(readings: list[LayerReading], builds: dict[int, BuildUp], cos: dict[int, dict[int, list[float]]]) -> str:
    refs = sorted(cos)
    lines = [
        "# Emotion vectors across depth (means over the 12 emotions)",
        "",
        "- attention / MLP: share of the layer's own |v| added by that layer's writes, along v.",
        "- mlp cos: cosine of the MLP's emotion write with the layer's vector (both denoised).",
        "- cos ref: cosine with the vector at a reference layer.",
        "",
        "| layer | |v| | attention | MLP | mlp cos | cos prev | " + " | ".join(f"cos {r}" for r in refs) + " | held-out acc |",
        "|---|---|---|---|---|---|" + "---|" * len(refs) + "---|",
    ]
    for r in readings:
        cells = [f"{mean(cos[ref][r.layer]):.2f}" if r.layer in cos[ref] else "" for ref in refs]
        lines.append(
            f"| {r.layer} | {mean(r.norm_denoised):.1f} | {mean(r.attention_along):+.3f} | {mean(r.mlp_along):+.3f} | "
            f"{mean(r.mlp_cos):.2f} | {mean(r.cos_previous):.2f} | " + " | ".join(cells) + f" | {r.heldout_accuracy:.2f} |"
        )
    for ref, b in builds.items():
        lines += ["", f"## Build-up of the layer-{ref} vector", "", f"- embeddings: {mean(b.embed):+.3f}"]
        lines.append(f"- all attention: {sum(mean(s) for s in b.attention):+.3f}; all MLPs: {sum(mean(s) for s in b.mlp):+.3f}")
        top = sorted(
            [(f"attention {i}", mean(s)) for i, s in enumerate(b.attention)]
            + [(f"MLP {i}", mean(s)) for i, s in enumerate(b.mlp)],
            key=lambda kv: -abs(kv[1]),
        )[:8]
        lines.append("- largest single components: " + ", ".join(f"{k} {v:+.3f}" for k, v in top))
    return "\n".join(lines) + "\n"


def main():
    cfg = load_run_config(sys.argv[1:])
    t0 = time.time()
    means = LayerMeans(cfg.run_dir)
    split = StorySplit.of(load_story_records(cfg.run_dir))
    readings, denoised = read_layers(means, split, cfg.vectors)
    last = means.n_layers - 1
    refs = (*REFERENCE_LAYERS, last)
    builds = {ref: build_up(means, split, denoised[ref], ref) for ref in refs}
    for ref, b in builds.items():
        total = [e + sum(a[i] for a in b.attention) + sum(m[i] for m in b.mlp) for i, e in enumerate(b.embed)]
        assert all(abs(t - 1) < 1e-6 for t in total), f"build-up of layer {ref} does not sum to 1: {total}"
    cos = {ref: cos_to(denoised, ref) for ref in refs}

    out = cfg.result_dir
    (out / "charts").mkdir(parents=True, exist_ok=True)
    payload = {
        "readings": [r.model_dump() for r in readings],
        "build_up": {ref: b.model_dump() for ref, b in builds.items()},
        "cos_to": {ref: {layer: c for layer, c in by.items()} for ref, by in cos.items()},
    }
    (out / "journey.json").write_text(json.dumps(payload))
    (out / "journey.md").write_text(markdown(readings, builds, cos))
    for name, svg in charts(readings, builds, cos).items():
        (out / "charts" / f"{name}.svg").write_text(svg)
    print(f"[saved] {out}/journey.json, journey.md, charts/")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
