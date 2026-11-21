"""Swap the value vectors (`W_down` columns) of two emotions' sparse sets, and read what changes. GPU.

    python scripts/swap_values.py --config configs/causal_sets.yaml

- For a pair e -> f: pair the neurons of S_e and S_f by greedy rank (shared neurons left out), and swap their
  `W_down` columns at layer 25. The activations are not touched: e's neurons still fire on e's stories, but now
  write f's values.
- Baseline: swap the columns of the same number of random neuron pairs.
- On e's held-out stories, the readouts of `causal_sets.py`: the score gap (f minus e) at later layers, the share
  labelled f, and the answer gap log p(f) - log p(e).
- Writes results/<run_id>/swap_values.json.
"""

import json
import random
import sys
import time

import torch
from safetensors.torch import load_file

from emotion_neurons.config import load_run_config
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.interventions import emotion_scores, run_with_edits
from emotion_neurons.models import decoder_layers, load_model
from emotion_neurons.texts import load_stories

torch.set_grad_enabled(False)


def greedy_pairs(order_e: list[int], order_f: list[int]) -> list[tuple[int, int]]:
    shared = set(order_e) & set(order_f)
    a = [n for n in order_e if n not in shared]
    b = [n for n in order_f if n not in shared]
    return list(zip(a, b))


def swap_columns(W: torch.Tensor, pairs: list[tuple[int, int]]) -> None:
    """In place; calling it twice restores W."""
    i = torch.tensor([p[0] for p in pairs], device=W.device)
    j = torch.tensor([p[1] for p in pairs], device=W.device)
    left, right = W[:, i].clone(), W[:, j].clone()
    W[:, i], W[:, j] = right, left


def main():
    cfg = load_run_config(sys.argv[1:])
    cs = cfg.causal_sets
    t0 = time.time()
    model, tokenizer = load_model(cfg.model, cfg.device)
    W = decoder_layers(model)[cfg.layer].mlp.down_proj.weight
    units = load_file(str(cs.layer_vectors_file))["unit"].float()[cs.read_layers]
    sets = {e: v["match"]["neurons"] for e, v in json.loads(cs.sets_file.read_text()).items()}
    answer_ids = [tokenizer(" " + e, add_special_tokens=False)["input_ids"][0] for e in EMOTIONS]
    stories = [s for s in load_stories(cfg.data) if s.heldout and s.emotion in EMOTIONS]
    rng = random.Random(cs.seed)

    def readouts(idx: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        means, answers = [], []
        for i in idx:
            n = len(tokenizer(stories[i].text)["input_ids"])
            ids = tokenizer(stories[i].text + cs.question, return_tensors="pt")["input_ids"].to(cfg.device)
            r = run_with_edits(model, ids, [], cs.read_layers, (cfg.data.split_token, n - 1))
            means.append(r.stream_means.cpu())
            answers.append(torch.log_softmax(r.last_logits[answer_ids], -1).cpu())
        return torch.stack(means), torch.stack(answers)

    # Centre: the clean mean over all held-out stories, as in causal_sets.py.
    all_clean, _ = readouts(list(range(len(stories))))
    centre = all_clean.mean(0)
    print(f"[clean] {len(stories)} stories, {time.time() - t0:.0f}s", flush=True)

    def scores(means: torch.Tensor) -> torch.Tensor:
        return torch.stack([emotion_scores(means[:, j], centre[j], units[j]) for j in range(len(cs.read_layers))], 1)

    out = {}
    for e_name, f_name in cs.swaps:
        e, f = EMOTIONS.index(e_name), EMOTIONS.index(f_name)
        idx = [i for i, s in enumerate(stories) if s.emotion == e_name]
        clean_m, clean_a = readouts(idx)
        clean_s = scores(clean_m)
        pairs = greedy_pairs(sets[e_name], sets[f_name])
        conditions = {"set": pairs}
        for r in range(cs.random_sets):
            picks = rng.sample(range(W.shape[1]), 2 * len(pairs))
            conditions[f"random{r}"] = list(zip(picks[: len(pairs)], picks[len(pairs) :]))
        row = {"pairs": len(pairs)}
        for name, p in conditions.items():
            swap_columns(W, p)
            m, a = readouts(idx)
            swap_columns(W, p)
            s = scores(m)
            row[name] = {
                "gap_change": {k: float(((s[:, j, f] - s[:, j, e]) - (clean_s[:, j, f] - clean_s[:, j, e])).mean()) for j, k in enumerate(cs.read_layers)},
                "clean_gap": {k: float((clean_s[:, j, e] - clean_s[:, j, f]).mean()) for j, k in enumerate(cs.read_layers)},
                "label_is_target": {k: float((s[:, j].argmax(-1) == f).float().mean()) for j, k in enumerate(cs.read_layers)},
                "answer_gap_change": float(((a[:, f] - a[:, e]) - (clean_a[:, f] - clean_a[:, e])).mean()),
                "answer_is_target": float((a.argmax(-1) == f).float().mean()),
            }  # fmt: skip
        out[f"{e_name}->{f_name}"] = row
        last = cs.read_layers[-1]
        rand = sum(row[f"random{r}"]["gap_change"][last] for r in range(cs.random_sets)) / cs.random_sets
        print(f"[{e_name}->{f_name}] {len(pairs)} pairs | L{last} gap change: set {row['set']['gap_change'][last]:+.1f}, random {rand:+.1f}, "
              f"clean gap {row['set']['clean_gap'][last]:.1f} | answer gap change: set {row['set']['answer_gap_change']:+.2f} | "
              f"label is target: set {row['set']['label_is_target'][last]:.2f}, {time.time() - t0:.0f}s", flush=True)  # fmt: skip
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "swap_values.json").write_text(json.dumps(out, indent=1))
    print(f"[saved] {cfg.result_dir}/swap_values.json")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
