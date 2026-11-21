"""Does each layer's emotion vector follow emotion words or the situation? GPU.

    python scripts/word_tests.py --config configs/word_tests.yaml

- Text sets, each balanced over the 12 emotions:
  - `heldout`: the 240 held-out training stories (reference);
  - `explicit`: adult texts that name the emotion (`data/adult_stories`, explicit = true);
  - `implied`: adult texts that do not name it (explicit = false);
  - `mislead`: texts whose character feels the target emotion, while a different "distractor" emotion word appears
    3 or more times, negated or about someone else (`data/word_tests/mislead.jsonl`).
- Per text and layer: the mean `resid_post` over tokens 50+, centred on the mean of its own set, projected on that
  layer's 12 unit emotion vectors. The label is the argmax.
- Per set and layer: accuracy; for `mislead` also the share labelled as the distractor, and the score gap
  true - distractor.
- The model's own answer, as in `causal_sets.py`: log-softmax over the 12 emotion words after a question.
- Writes results/<run_id>/word_tests.json.
"""

import json
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch
from pydantic import BaseModel
from safetensors.torch import load_file

from emotion_neurons.config import load_run_config
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.models import decoder_layers, load_model
from emotion_neurons.models.qwen3 import hooks
from emotion_neurons.texts import load_labelled_texts, load_stories
from emotion_neurons.texts.stories import read_jsonl

torch.set_grad_enabled(False)


class TestText(BaseModel):
    id: str
    emotion: str
    text: str
    distractor: str | None = None


def text_sets(cfg) -> dict[str, list[TestText]]:
    adult = load_labelled_texts(cfg.word_tests.adult_dir)
    return {
        "heldout": [TestText(id=s.id, emotion=s.emotion, text=s.text) for s in load_stories(cfg.data) if s.heldout and s.emotion in EMOTIONS],
        "explicit": [TestText(id=t.id, emotion=t.emotion, text=t.text) for t in adult if t.explicit],
        "implied": [TestText(id=t.id, emotion=t.emotion, text=t.text) for t in adult if not t.explicit],
        "mislead": [TestText(**row) for row in read_jsonl(cfg.word_tests.mislead_file)],
    }  # fmt: skip


def label_words(text: str) -> list[str]:
    """The 12 emotion labels that appear as whole words (a check that `implied` texts really avoid them)."""
    return [e for e in EMOTIONS if re.search(rf"\b{e}\b", text.lower())]


@torch.no_grad()
def layer_means(model, input_ids, start: int, stop: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean resid_post over [start, stop) at every layer, and the last-token logits."""
    means: dict[int, torch.Tensor] = {}

    def keep(layer: int) -> Callable:
        return lambda _m, _a, out: means.__setitem__(
            layer, (out[0] if isinstance(out, tuple) else out)[0, start:stop].float().mean(0)
        )

    blocks = decoder_layers(model)
    with hooks([b.register_forward_hook(keep(k)) for k, b in enumerate(blocks)]):
        logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
    return torch.stack([means[k] for k in range(len(blocks))]).cpu(), logits


@torch.no_grad()
def token_streams(model, input_ids) -> torch.Tensor:
    """resid_post at every layer and token: [layer, token, d_model]."""
    out: dict[int, torch.Tensor] = {}
    blocks = decoder_layers(model)
    keep = [
        b.register_forward_hook(lambda _m, _a, o, k=k: out.__setitem__(k, (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))
        for k, b in enumerate(blocks)
    ]
    with hooks(keep):
        model.model(input_ids=input_ids, use_cache=False)
    return torch.stack([out[k] for k in range(len(blocks))])


def distractor_tokens(model, tokenizer, texts: list[TestText], units: torch.Tensor) -> dict:
    """At each token of the distractor word (e.g. " afraid" in "was not afraid"): which emotion each layer reads there.

    - Centred on the mean over all tokens 4+ of these texts, per layer.
    - Lexical reading: the label is the distractor. Contextual reading: it moves to the target, or away from the word.
    """
    streams, positions = [], []
    for t in texts:
        enc = tokenizer(t.text, return_offsets_mapping=True, return_tensors="pt")
        spans = [m.span() for m in re.finditer(rf"\b{t.distractor}\b", t.text, flags=re.IGNORECASE)]
        pos = [i for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist()) if any(a < e and b > s0 for s0, e in spans)]
        streams.append(token_streams(model, enc["input_ids"].to(model.device)))
        positions.append(pos)
    centre = torch.cat([s[:, 4:] for s in streams], 1).mean(1)  # [layer, d_model]
    rows = []
    for t, s, pos in zip(texts, streams, positions):
        for p in pos:
            scores = torch.einsum("ld,led->le", s[:, p] - centre, units)
            rows.append((EMOTIONS.index(t.distractor), EMOTIONS.index(t.emotion), scores))
    d = torch.tensor([r[0] for r in rows])
    y = torch.tensor([r[1] for r in rows])
    sc = torch.stack([r[2] for r in rows])  # [token, layer, emotion]
    idx = torch.arange(len(rows))
    return {
        "n_tokens": len(rows),
        "labelled_distractor": (sc.argmax(-1) == d[:, None]).float().mean(0).tolist(),
        "labelled_target": (sc.argmax(-1) == y[:, None]).float().mean(0).tolist(),
        "distractor_rank": (sc > sc[idx, :, d][..., None]).sum(-1).float().mean(0).tolist(),
    }


def summarise(texts: list[TestText], means: torch.Tensor, answers: torch.Tensor, units: torch.Tensor) -> dict:
    labels = torch.tensor([EMOTIONS.index(t.emotion) for t in texts])
    scores = torch.einsum("tld,led->tle", means - means.mean(0), units)  # [text, layer, emotion]
    out = {
        "n": len(texts),
        "accuracy": (scores.argmax(-1) == labels[:, None]).float().mean(0).tolist(),
        "answer_accuracy": float((answers.argmax(-1) == labels).float().mean()),
        "texts_with_label_words": sum(bool(label_words(t.text)) for t in texts),
    }
    if texts[0].distractor:
        d = torch.tensor([EMOTIONS.index(t.distractor) for t in texts])
        idx = torch.arange(len(texts))
        out["labelled_distractor"] = (scores.argmax(-1) == d[:, None]).float().mean(0).tolist()
        out["gap_true_minus_distractor"] = (scores[idx, :, labels] - scores[idx, :, d]).mean(0).tolist()
        out["answer_is_distractor"] = float((answers.argmax(-1) == d).float().mean())
    return out


def main():
    cfg = load_run_config(sys.argv[1:])
    wt = cfg.word_tests
    t0 = time.time()
    model, tokenizer = load_model(cfg.model, cfg.device)
    units = load_file(str(wt.layer_vectors_file))["unit"].float()  # [layer, emotion, d_model]
    answer_ids = [tokenizer(" " + e, add_special_tokens=False)["input_ids"][0] for e in EMOTIONS]
    out = {}
    for name, texts in text_sets(cfg).items():
        means, answers = [], []
        for t in texts:
            n = len(tokenizer(t.text)["input_ids"])
            ids = tokenizer(t.text + cfg.causal_sets.question, return_tensors="pt")["input_ids"].to(cfg.device)
            m, logits = layer_means(model, ids, cfg.data.split_token, n - 1)
            means.append(m)
            answers.append(torch.log_softmax(logits[answer_ids], -1).cpu())
        out[name] = summarise(texts, torch.stack(means), torch.stack(answers), units)
        acc = out[name]["accuracy"]
        extra = (
            f" | distractor {[round(x, 2) for x in out[name]['labelled_distractor'][::5]]}"
            if "labelled_distractor" in out[name]
            else ""
        )
        print(
            f"[{name}] n {len(texts)} acc every 5th layer {[round(x, 2) for x in acc[::5]]} answer {out[name]['answer_accuracy']:.2f}{extra}, {time.time() - t0:.0f}s",
            flush=True,
        )
    mislead = text_sets(cfg)["mislead"]
    out["distractor_tokens"] = distractor_tokens(model, tokenizer, mislead, units)
    dt = out["distractor_tokens"]
    print(f"[distractor tokens] n {dt['n_tokens']} labelled distractor every 5th layer {[round(x, 2) for x in dt['labelled_distractor'][::5]]} "
          f"target {[round(x, 2) for x in dt['labelled_target'][::5]]}", flush=True)  # fmt: skip
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.result_dir / "word_tests.json").write_text(json.dumps(out, indent=1))
    print(f"[saved] {cfg.result_dir}/word_tests.json")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
