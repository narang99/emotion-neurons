"""Does an emotion's MLP write come from tail firings (above p97, below p3) or from small shifts on every token? GPU.

    python scripts/tail_check.py --config configs/tail_check.yaml

- Thresholds: p3 and p97 of every layer-25 neuron over a sample of the general harvest corpus (tokens 4+).
- Shift split (exact): on the train stories, tokens 50+, each token of each neuron is in one class:
  `high` (>= p97), `low` (<= p3) or `middle`.
  - `dA_e[n] = sum over classes of dA_e^class[n]`, where `dA_e^class` is the same mean contrast as `dA_e`
    (story mean, then emotion mean, minus the 12-emotion mean) of `a * 1[class]`.
  - A check asserts that the classes add up to Stage 2's `dA_e`.
- Per emotion, for its sparse set S_e and for all neurons:
  - the share of the write along `delta_e` that each class carries: `(W_down[:, S] dA_e^class[S]) . delta_e / |delta_e|^2`;
  - the share of story tokens in each class.
- Writes results/<run_id>/tail_check.json.
"""

import json
import sys
import time

import torch
from safetensors.torch import load_file

from emotion_neurons.config import load_run_config
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.harvest import collect_windows, padded_batch
from emotion_neurons.models import decoder_layers, load_model, read_mlp
from emotion_neurons.texts import load_stories

torch.set_grad_enabled(False)
CLASSES = ("high", "low", "middle")


def corpus_thresholds(model, tokenizer, cfg) -> tuple[torch.Tensor, torch.Tensor, int]:
    """p3 and p97 per neuron over the corpus sample."""
    q, n = corpus_quantiles(model, tokenizer, cfg, [0.03, 0.97])
    return q[0], q[1], n


def corpus_quantiles(model, tokenizer, cfg, quantiles: list[float]) -> tuple[torch.Tensor, int]:
    """Per neuron, the quantiles over the corpus sample [quantile, neuron], on the GPU in chunks of neurons.

    - Quantile q is the k-th smallest value with k = int(q * n_tokens), as `kthvalue` in the first version.
    """
    h = cfg.harvest
    windows = collect_windows(h, tokenizer)
    all_neurons = torch.arange(model.config.intermediate_size, device=model.device)
    rows = []
    for start in range(0, len(windows.tokens), h.batch_size):
        ids, mask = padded_batch(windows.tokens[start : start + h.batch_size], tokenizer.pad_token_id)
        _, a = read_mlp(model, cfg.layer, ids.to(model.device), mask.to(model.device), all_neurons)
        keep = mask.bool().to(a.device)
        keep[:, : h.skip_first] = False
        rows.append(a[keep].half().cpu())
    acts = torch.cat(rows)  # [token, neuron]
    ranks = torch.tensor([max(1, int(q * len(acts))) - 1 for q in quantiles], device=model.device)
    out = torch.empty(len(quantiles), acts.shape[1])
    for c in range(0, acts.shape[1], 512):
        chunk = acts[:, c : c + 512].to(model.device).float()
        out[:, c : c + 512] = chunk.sort(0).values[ranks].cpu()
    return out, len(acts)


def story_class_means(model, tokenizer, stories, layer: int, lo, hi, split_token: int) -> tuple[dict, torch.Tensor]:
    """Per class, each story's mean over tokens 50+ of `a * 1[class]` [story, neuron]; and its token share per neuron."""
    out = {c: [] for c in CLASSES}
    shares = []
    lo_d, hi_d = lo.to(model.device), hi.to(model.device)
    for s in stories:
        ids = tokenizer(s.text, return_tensors="pt")["input_ids"].to(model.device)
        _, a = read_mlp(
            model, layer, ids, torch.ones_like(ids), torch.arange(model.config.intermediate_size, device=model.device)
        )
        a = a[0, split_token:]
        masks = {"high": a >= hi_d, "low": a <= lo_d}
        masks["middle"] = ~(masks["high"] | masks["low"])
        for c in CLASSES:
            out[c].append((a * masks[c]).mean(0).cpu())
        shares.append(torch.stack([masks[c].float().mean(0).cpu() for c in CLASSES]))  # [class, neuron]
    return {c: torch.stack(v) for c, v in out.items()}, torch.stack(shares)


def contrast(x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Emotion mean minus the 12-emotion mean, as Stage 2 builds dA."""
    means = torch.stack([x[labels == e].mean(0) for e in range(len(EMOTIONS))])
    return means - means.mean(0)


def main():
    cfg = load_run_config(sys.argv[1:])
    tc = cfg.tail_check
    t0 = time.time()
    model, tokenizer = load_model(cfg.model, cfg.device)
    lo, hi, n_tokens = corpus_thresholds(model, tokenizer, cfg)
    print(f"[thresholds] p3/p97 of {len(lo)} neurons over {n_tokens} corpus tokens, {time.time() - t0:.0f}s", flush=True)
    stories = [s for s in load_stories(cfg.data) if not s.heldout and s.emotion in EMOTIONS]
    labels = torch.tensor([EMOTIONS.index(s.emotion) for s in stories])
    class_means, token_shares = story_class_means(model, tokenizer, stories, cfg.layer, lo, hi, cfg.data.split_token)
    print(f"[stories] {len(stories)} train stories, {time.time() - t0:.0f}s", flush=True)

    vectors = load_file(str(tc.vectors_file))
    dA = vectors["mlp_act/mean/raw"].float()
    parts = {c: contrast(class_means[c].double(), labels).float() for c in CLASSES}
    gap = float((sum(parts.values()) - dA).abs().max() / dA.abs().max())
    print(f"[check] classes sum to Stage 2's dA within {gap:.1e} (relative)", flush=True)
    assert gap < 0.05, "the class split does not rebuild dA; the story set or token range differs from Stage 2"

    W_down = decoder_layers(model)[cfg.layer].mlp.down_proj.weight.float().cpu()
    sets = json.loads(tc.sets_file.read_text())
    out = {"corpus_tokens": n_tokens, "rebuild_gap": gap, "emotions": {}}
    for e, name in enumerate(EMOTIONS):
        write = W_down @ dA[e]
        S = torch.tensor(sets[name]["match"]["neurons"])
        row = {"set_size": len(S)}
        for scope, neurons in (("set", S), ("all", torch.arange(dA.shape[1]))):
            row[scope] = {c: float((W_down[:, neurons] @ parts[c][e, neurons]) @ write / write.norm() ** 2) for c in CLASSES}
        row["token_shares_set"] = dict(zip(CLASSES, token_shares[labels == e][:, :, S].mean((0, 2)).tolist()))
        row["token_shares_all"] = dict(zip(CLASSES, token_shares[labels == e].mean((0, 2)).tolist()))
        out["emotions"][name] = row
        print(f"[{name}] set {len(S)}: write share high {row['set']['high']:+.2f} low {row['set']['low']:+.2f} middle {row['set']['middle']:+.2f} "
              f"| set tokens high {row['token_shares_set']['high']:.3f} low {row['token_shares_set']['low']:.3f} | all neurons: high {row['all']['high']:+.2f} low {row['all']['low']:+.2f} middle {row['all']['middle']:+.2f}", flush=True)  # fmt: skip
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "tail_check.json").write_text(json.dumps(out, indent=1))
    print(f"[saved] {cfg.result_dir}/tail_check.json")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
