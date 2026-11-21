"""Causal tail test: does the model use the tail firings of the sparse-set neurons? GPU.

    python scripts/tail_causal.py --config configs/tail_causal.yaml

- Thresholds: p3 and p97 of every layer-25 neuron over a sample of the general corpus (as `tail_check.py`).
- On each held-out story of emotion e, at every token of layer 25:
  - `clip`: clamp the neurons of S into [p3, p97]. Only the tail firings change; the middle is left as it is;
  - `remove`: `a[S] -= dA_e[S]`, the mean shift (as `causal_sets.py`), for comparison;
  - S is the emotion's sparse set, 3 random sets of the same size, or all neurons.
- Readouts: the true emotion's score as a share of clean, the accuracy at later layers, and the model's answer.
- Writes results/<run_id>/tail_causal.json.
"""

import json
import random
import sys
import time

import torch
from safetensors.torch import load_file

from emotion_neurons.config import load_run_config
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.interventions import NeuronEdit, emotion_scores, run_with_edits
from emotion_neurons.models import load_model
from emotion_neurons.texts import load_stories

torch.set_grad_enabled(False)
sys.path.insert(0, "scripts")
from tail_check import corpus_thresholds  # noqa: E402


def main():
    cfg = load_run_config(sys.argv[1:])
    cs = cfg.causal_sets
    t0 = time.time()
    model, tokenizer = load_model(cfg.model, cfg.device)
    lo, hi, n_tokens = corpus_thresholds(model, tokenizer, cfg)
    print(f"[thresholds] {n_tokens} corpus tokens, {time.time() - t0:.0f}s", flush=True)
    shifts = load_file(str(cs.vectors_file))["mlp_act/mean/raw"].float()
    units = load_file(str(cs.layer_vectors_file))["unit"].float()[cs.read_layers]
    sets = {e: torch.tensor(sorted(v["match"]["neurons"])) for e, v in json.loads(cs.sets_file.read_text()).items()}
    rng = random.Random(cs.seed)
    d_mlp = shifts.shape[1]
    randoms = {e: [torch.tensor(sorted(rng.sample(range(d_mlp), len(sets[e])))) for _ in range(cs.random_sets)] for e in EMOTIONS}
    everyone = torch.arange(d_mlp)
    answer_ids = [tokenizer(" " + e, add_special_tokens=False)["input_ids"][0] for e in EMOTIONS]
    stories = [s for s in load_stories(cfg.data) if s.heldout and s.emotion in EMOTIONS]

    def clip(S):
        return [NeuronEdit(layer=cfg.layer, neurons=S, values=lo[S], upper=hi[S], kind="clip")]

    def remove(S, e):
        return [NeuronEdit(layer=cfg.layer, neurons=S, values=-shifts[e][S], kind="add")]

    streams: dict[str, list] = {}
    answers: dict[str, list] = {}
    for i, s in enumerate(stories):
        e = EMOTIONS.index(s.emotion)
        conditions = {
            "clean": [],
            "clip/set": clip(sets[s.emotion]),
            "clip/all": clip(everyone),
            "remove/set": remove(sets[s.emotion], e),
        }
        for r, R in enumerate(randoms[s.emotion]):
            conditions[f"clip/random{r}"] = clip(R)
        n = len(tokenizer(s.text)["input_ids"])
        ids = tokenizer(s.text + cs.question, return_tensors="pt")["input_ids"].to(cfg.device)
        for name, edits in conditions.items():
            out = run_with_edits(model, ids, edits, cs.read_layers, (cfg.data.split_token, n - 1))
            streams.setdefault(name, []).append(out.stream_means.cpu())
            answers.setdefault(name, []).append(torch.log_softmax(out.last_logits[answer_ids], -1).cpu())
        if i % 60 == 59:
            print(f"[stories] {i + 1}/{len(stories)}, {time.time() - t0:.0f}s", flush=True)

    labels = torch.tensor([EMOTIONS.index(s.emotion) for s in stories])
    idx = torch.arange(len(stories))
    centre = torch.stack(streams["clean"]).mean(0)
    scores = {
        k: torch.stack([emotion_scores(torch.stack(v)[:, j], centre[j], units[j]) for j in range(len(cs.read_layers))], 1)
        for k, v in streams.items()
    }
    clean_true = scores["clean"][idx, :, labels]
    result = {}
    for name, s in scores.items():
        a = torch.stack(answers[name])
        result[name] = {
            "true_score_vs_clean": {k: float(s[idx, j, labels].mean() / clean_true[:, j].mean()) for j, k in enumerate(cs.read_layers)},
            "accuracy": {k: float((s[:, j].argmax(-1) == labels).float().mean()) for j, k in enumerate(cs.read_layers)},
            "answer_logp_true": float(a[idx, labels].mean()),
            "answer_accuracy": float((a.argmax(-1) == labels).float().mean()),
        }  # fmt: skip
        r = result[name]
        print(f"  {name:14s} true score vs clean {[round(v, 3) for v in r['true_score_vs_clean'].values()]} acc {[round(v, 2) for v in r['accuracy'].values()]} "
              f"answer logp {r['answer_logp_true']:.3f}", flush=True)  # fmt: skip
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "tail_causal.json").write_text(json.dumps({"read_layers": cs.read_layers, "conditions": result}, indent=1))
    print(f"[saved] {cfg.result_dir}/tail_causal.json")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
