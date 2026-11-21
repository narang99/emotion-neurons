"""Causal test of the layer-25 sparse sets: remove or swap an emotion's activation shift, read what changes. GPU.

    python scripts/causal_sets.py --config configs/causal_sets.yaml

- Sets: each emotion's sparse set S_e that matches the full MLP on held-out stories (`sets.json`, `match`).
- Shifts: `dA_e`, each neuron's mean activation on emotion e's train stories minus its 12-emotion mean.
- On each held-out story of emotion e (20 per emotion), at every token of layer 25:
  - removal: `a[S] -= dA_e[S]`, for S = S_e, random sets of the same size, and all neurons;
  - mean ablation: `a[S] = 12-emotion mean`, for S = S_e and the random sets;
  - swap e -> f (chosen pairs, on e's stories): `a[S] += scale * (dA_f - dA_e)[S]`, for S = S_e | S_f, random sets
    of that size, and all neurons.
- Readouts of each pass:
  - emotion scores at later layers: the mean `resid_post` over story tokens 50+, centred on the clean mean of all
    held-out stories, projected on that layer's unit emotion vectors; the label is the argmax;
  - the model's answer: after the story, "In one word, how does the main character feel? ... feels", the log-softmax
    over the 12 emotion words at the last token.
- A few greedy continuations per swap pair, for reading.
- Writes results/<run_id>/causal_sets.json (summary and continuations) and runs/<run_id>/readouts.safetensors.
"""

import json
import random
import sys
import time

import torch
from safetensors.torch import load_file, save_file

from emotion_neurons.config import load_run_config
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.interventions import NeuronEdit, emotion_scores, generate_with_edits, run_with_edits
from emotion_neurons.models import decoder_layers, load_model
from emotion_neurons.texts import load_stories

torch.set_grad_enabled(False)


def answer_token_ids(tokenizer) -> tuple[list[int], list[bool]]:
    """First token of " <emotion>" for each emotion, and whether the word is a single token."""
    pieces = [tokenizer(" " + e, add_special_tokens=False)["input_ids"] for e in EMOTIONS]
    return [p[0] for p in pieces], [len(p) == 1 for p in pieces]


def random_sets(size: int, count: int, d_mlp: int, rng: random.Random) -> list[torch.Tensor]:
    return [torch.tensor(sorted(rng.sample(range(d_mlp), size))) for _ in range(count)]


def write_share(W_down: torch.Tensor, shift: torch.Tensor, neurons: torch.Tensor, full_write: torch.Tensor) -> float:
    """Share of the emotion write, along itself, that the neurons' part of the shift carries."""
    part = W_down[:, neurons] @ shift[neurons]
    return float(part @ full_write / full_write.norm() ** 2)


class Conditions:
    """Builds the edit lists of one story's conditions, by name."""

    def __init__(
        self,
        layer: int,
        shifts: torch.Tensor,
        grand_mean: torch.Tensor,
        sets: dict[str, torch.Tensor],
        randoms: dict[str, list[torch.Tensor]],
    ):
        self.layer, self.shifts, self.grand_mean, self.sets, self.randoms = layer, shifts, grand_mean, sets, randoms
        self.all = torch.arange(shifts.shape[1])

    def edit(self, neurons: torch.Tensor, values: torch.Tensor, kind: str) -> list[NeuronEdit]:
        return [NeuronEdit(layer=self.layer, neurons=neurons, values=values, kind=kind)]

    def removal(self, e: int) -> dict[str, list[NeuronEdit]]:
        name, shift = EMOTIONS[e], -self.shifts[e]
        out = {"clean": [], "remove/set": self.edit(self.sets[name], shift[self.sets[name]], "add"),
               "remove/all": self.edit(self.all, shift, "add"),
               "mean/set": self.edit(self.sets[name], self.grand_mean[self.sets[name]], "hold")}  # fmt: skip
        for i, r in enumerate(self.randoms[name]):
            out[f"remove/random{i}"] = self.edit(r, shift[r], "add")
            out[f"mean/random{i}"] = self.edit(r, self.grand_mean[r], "hold")
        return out

    def swap(self, e: int, f: int, scale: float, randoms: list[torch.Tensor]) -> dict[str, list[NeuronEdit]]:
        shift = scale * (self.shifts[f] - self.shifts[e])
        union = torch.unique(torch.cat([self.sets[EMOTIONS[e]], self.sets[EMOTIONS[f]]]))
        out = {"set": self.edit(union, shift[union], "add"), "all": self.edit(self.all, shift, "add")}
        for i, r in enumerate(randoms):
            out[f"random{i}"] = self.edit(r, shift[r], "add")
        return out


def summarise_removal(scores: dict, answers: dict, labels: torch.Tensor, read_layers: list[int]) -> dict:
    """Per condition: accuracy at each layer, the true emotion's score as a share of clean, answer accuracy and logp."""
    out = {}
    idx = torch.arange(len(labels))
    clean_true = scores["clean"][idx, :, labels]  # [story, layer]
    for name, s in scores.items():
        true = s[idx, :, labels]
        out[name] = {
            "accuracy": {k: float((s[:, j].argmax(-1) == labels).float().mean()) for j, k in enumerate(read_layers)},
            "true_score_vs_clean": {k: float(true[:, j].mean() / clean_true[:, j].mean()) for j, k in enumerate(read_layers)},
            "answer_accuracy": float((answers[name].argmax(-1) == labels).float().mean()),
            "answer_logp_true": float(answers[name][idx, labels].mean()),
        }
    return out


def mean_over_randoms(summary: dict, prefix: str) -> dict:
    """Mean and range of the random sets' numbers, as one entry."""
    runs = [v for k, v in summary.items() if k.startswith(prefix + "random")]
    if not runs:
        return {}

    def combine(values):
        return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}

    out = {}
    for key, value in runs[0].items():
        if isinstance(value, dict):
            out[key] = {k: combine([r[key][k] for r in runs]) for k in value}
        else:
            out[key] = combine([r[key] for r in runs])
    return out


def main():
    cfg = load_run_config(sys.argv[1:])
    cs = cfg.causal_sets
    t0 = time.time()
    model, tokenizer = load_model(cfg.model, cfg.device)
    d_mlp = model.config.intermediate_size
    vectors = load_file(str(cs.vectors_file))
    shifts, grand_mean = vectors["mlp_act/mean/raw"].float(), vectors["mlp_act/grand_mean"].float()
    full_writes = vectors["mlp_out/mean/raw"].float()
    units = load_file(str(cs.layer_vectors_file))["unit"].float()[cs.read_layers]  # [layer, emotion, d_model]
    sets_json = json.loads(cs.sets_file.read_text())
    sets = {e: torch.tensor(sorted(sets_json[e]["match"]["neurons"])) for e in EMOTIONS}
    rng = random.Random(cs.seed)
    randoms = {e: random_sets(len(sets[e]), cs.random_sets, d_mlp, rng) for e in EMOTIONS}
    W_down = decoder_layers(model)[cfg.layer].mlp.down_proj.weight.float().cpu()
    conditions = Conditions(cfg.layer, shifts, grand_mean, sets, randoms)
    answer_ids, single = answer_token_ids(tokenizer)

    stories = [s for s in load_stories(cfg.data) if s.heldout and s.emotion in EMOTIONS]
    labels = torch.tensor([EMOTIONS.index(s.emotion) for s in stories])
    prompts, ranges = [], []
    for s in stories:
        n_story = len(tokenizer(s.text)["input_ids"])
        prompts.append(tokenizer(s.text + cs.question, return_tensors="pt")["input_ids"].to(cfg.device))
        ranges.append((cfg.data.split_token, n_story - 1))
    print(
        f"[load] {len(stories)} stories, set sizes {[len(sets[e]) for e in EMOTIONS]}, single-token answers {single}, {time.time() - t0:.0f}s",
        flush=True,
    )

    def read(i: int, edits: list[NeuronEdit]) -> tuple[torch.Tensor, torch.Tensor]:
        r = run_with_edits(model, prompts[i], edits, cs.read_layers, ranges[i])
        return r.stream_means.cpu(), torch.log_softmax(r.last_logits[answer_ids], -1).cpu()

    # Removal and mean ablation, on every held-out story.
    streams: dict[str, list] = {}
    answers: dict[str, list] = {}
    for i, s in enumerate(stories):
        for name, edits in conditions.removal(int(labels[i])).items():
            m, a = read(i, edits)
            streams.setdefault(name, []).append(m)
            answers.setdefault(name, []).append(a)
        if i % 40 == 39:
            print(f"[removal] {i + 1}/{len(stories)} stories, {time.time() - t0:.0f}s", flush=True)
    stream_t = {k: torch.stack(v) for k, v in streams.items()}  # [story, layer, d_model]
    centre = stream_t["clean"].mean(0)  # [layer, d_model]
    score = {
        k: torch.stack([emotion_scores(v[:, j], centre[j], units[j]) for j in range(len(cs.read_layers))], 1)
        for k, v in stream_t.items()
    }
    answer_t = {k: torch.stack(v) for k, v in answers.items()}
    removal = summarise_removal(score, answer_t, labels, cs.read_layers)
    removal["remove/random"] = mean_over_randoms(removal, "remove/")
    removal["mean/random"] = mean_over_randoms(removal, "mean/")
    shares = {e: {"set": write_share(W_down, shifts[k], sets[e], full_writes[k]),
                  "random": [write_share(W_down, shifts[k], r, full_writes[k]) for r in randoms[e]],
                  "set_size": len(sets[e])} for k, e in enumerate(EMOTIONS)}  # fmt: skip
    print(f"[removal] done, {time.time() - t0:.0f}s", flush=True)
    for name in ["clean", "remove/set", "remove/random", "remove/all", "mean/set", "mean/random"]:
        r = removal[name]
        acc = (
            r["accuracy"]
            if name != "remove/random" and name != "mean/random"
            else {k: v["mean"] for k, v in r["accuracy"].items()}
        )
        ans = r["answer_accuracy"] if isinstance(r["answer_accuracy"], float) else r["answer_accuracy"]["mean"]
        print(f"  {name:14s} acc {acc} answer acc {ans:.2f}", flush=True)

    # Swaps.
    swaps = {}
    generations = []
    for e_name, f_name in cs.swaps:
        e, f = EMOTIONS.index(e_name), EMOTIONS.index(f_name)
        idx = [i for i in range(len(stories)) if int(labels[i]) == e]
        union_size = len(torch.unique(torch.cat([sets[e_name], sets[f_name]])))
        swap_randoms = random_sets(union_size, cs.random_sets, d_mlp, rng)
        for scale in cs.swap_scales:
            key = f"{e_name}->{f_name} x{scale:g}"
            per_condition = {}
            clean_m = torch.stack([stream_t["clean"][i] for i in idx])
            clean_s = torch.stack([emotion_scores(clean_m[:, j], centre[j], units[j]) for j in range(len(cs.read_layers))], 1)
            clean_a = torch.stack([answer_t["clean"][i] for i in idx])
            for name in ["set", "all"] + [f"random{i}" for i in range(cs.random_sets)]:
                ms, ans = zip(*[read(i, conditions.swap(e, f, scale, swap_randoms)[name]) for i in idx])
                m, a = torch.stack(ms), torch.stack(ans)
                sc = torch.stack([emotion_scores(m[:, j], centre[j], units[j]) for j in range(len(cs.read_layers))], 1)
                per_condition[name] = {
                    "gap_change": {k: float(((sc[:, j, f] - sc[:, j, e]) - (clean_s[:, j, f] - clean_s[:, j, e])).mean()) for j, k in enumerate(cs.read_layers)},
                    "label_is_target": {k: float((sc[:, j].argmax(-1) == f).float().mean()) for j, k in enumerate(cs.read_layers)},
                    "answer_gap_change": float(((a[:, f] - a[:, e]) - (clean_a[:, f] - clean_a[:, e])).mean()),
                    "answer_is_target": float((a.argmax(-1) == f).float().mean()),
                    "answer_is_source": float((a.argmax(-1) == e).float().mean()),
                }  # fmt: skip
            per_condition["clean"] = {
                "label_is_target": {k: float((clean_s[:, j].argmax(-1) == f).float().mean()) for j, k in enumerate(cs.read_layers)},
                "answer_is_target": float((clean_a.argmax(-1) == f).float().mean()),
                "answer_is_source": float((clean_a.argmax(-1) == e).float().mean()),
            }  # fmt: skip
            per_condition["random"] = mean_over_randoms(per_condition, "")
            swaps[key] = {"union_size": union_size, "conditions": per_condition}
            p = per_condition
            print(f"[swap {key}] answer is target: clean {p['clean']['answer_is_target']:.2f} set {p['set']['answer_is_target']:.2f} "
                  f"random {p['random']['answer_is_target']['mean']:.2f} all {p['all']['answer_is_target']:.2f} | "
                  f"L{cs.read_layers[-1]} label is target: set {p['set']['label_is_target'][cs.read_layers[-1]]:.2f} all {p['all']['label_is_target'][cs.read_layers[-1]]:.2f}, {time.time() - t0:.0f}s", flush=True)  # fmt: skip
        for i in idx[: cs.generate_per_pair]:
            ids = tokenizer(stories[i].text + cs.continuation, return_tensors="pt")["input_ids"].to(cfg.device)
            scale = max(cs.swap_scales)
            edits = conditions.swap(e, f, scale, swap_randoms)
            row = {"story": stories[i].id, "swap": f"{e_name}->{f_name} x{scale:g}"}
            for name, ed in [("clean", []), ("set", edits["set"]), ("random0", edits["random0"]), ("all", edits["all"])]:
                row[name] = tokenizer.decode(generate_with_edits(model, ids, ed, cs.new_tokens))
            generations.append(row)

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    out = {"read_layers": cs.read_layers, "set_sizes": {e: len(sets[e]) for e in EMOTIONS}, "single_token_answers": dict(zip(EMOTIONS, single)),
           "write_shares": shares, "removal": removal, "swaps": swaps, "generations": generations}  # fmt: skip
    (cfg.result_dir / "causal_sets.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    save_file({f"{k}/streams": v.contiguous() for k, v in stream_t.items()} | {f"{k}/answers": v.contiguous() for k, v in answer_t.items()},
              str(cfg.run_dir / "readouts.safetensors"))  # fmt: skip
    print(f"[saved] {cfg.result_dir}/causal_sets.json")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
