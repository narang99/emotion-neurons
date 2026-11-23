"""Emotion vectors, as in the emotions paper, in three steps.

- capture (GPU): one pass over every story, pooled sums of the layer's sites.
  - runs/<run_id>/sums.safetensors: per site and range (early, late), an fp32 [story, dim] sum over that range's
    real tokens, plus `counts` [story, 2]. Late: the example set's positions (token 50 on); early: the rest.
  - runs/<run_id>/stories.json records the story order (`emotion_neurons.texts.load_stories`).
  - Checks: the most padded story, run alone, gives the same sums; the counted tokens equal the tokenized tokens.
- build (CPU): the vectors from the late-token means, by `vectors.recipe`.
  - results/<run_id>/vectors.safetensors (see `emotion_neurons.vectors.EmotionVectors`);
  - results/<run_id>/vectors_meta.json: emotions, sites, PCs removed per site, counts per split.
- check (CPU): results/<run_id>/checks.json: nearest-vector labels for every story, split-half cosine, emotion x
  emotion cosine.
- select does nothing. tokens (GPU) and report (CPU) write token_scores.* and report.html.
"""

import json
import subprocess
import time
from functools import cached_property

import torch
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..capture import late_means, padded, pooled_position_sums, position_mask
from ..config import EmotionVectorsExtraction, ExtractionRunConfig
from ..datasets import Rendered, example_set
from ..emotions import EMOTIONS, SITES
from ..models import adapter_for, read_layer
from ..reports import build_emotion_vectors_report, write_report
from ..texts import Story, StoryRecord, StorySplit, load_stories, load_story_records
from ..token_scores import save_token_scores, story_token_scores
from ..vectors import RECIPES, EmotionVectors, cosine_matrix, remove
from ..vectors.checks import confusion, nearest_vector_scores, split_half_cosine


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"


def batch_sums(model, layer: int, batch: list[Rendered], pad_id: int):
    ids, mask = padded([r.input_ids for r in batch], pad_id)
    late = position_mask([r.positions for r in batch], ids.shape[1])
    ids, mask, late = ids.to(model.device), mask.to(model.device), late.to(model.device)
    return pooled_position_sums(read_layer(model, layer, ids, mask), mask, late)


def pad_tokens_per_story(lengths: list[int], order: list[int], batch_size: int) -> dict[int, int]:
    """Pad tokens each story gets: its batch's longest story minus its own length. Batches follow `order`."""
    pad = {}
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        longest = max(lengths[i] for i in batch)
        pad.update({i: longest - lengths[i] for i in batch})
    return pad


def padding_check(model, pad_id: int, cfg: ExtractionRunConfig, stories, rendered, order, sums, counts) -> None:
    """The most padded story, run alone, must give the batched sums: padding must not leak into them."""
    extract = cfg.emotion_vectors.extract
    pad = pad_tokens_per_story([len(r.input_ids) for r in rendered], order, extract.batch_size)
    i = max(pad, key=pad.get)
    print(f"[padding check] {stories[i].id} had {pad[i]} pad tokens in its batch")
    alone, c = batch_sums(model, extract.layer, [rendered[i]], pad_id)
    assert torch.equal(c.cpu()[0], counts[i]), "token counts differ alone vs batched"
    for site, v in alone.items():
        rel = ((v.cpu()[0] - sums[site][i]).norm() / sums[site][i].norm()).item()
        print(f"[padding check] {stories[i].id} {site}: rel diff alone vs batched {rel:.2e}")
        assert rel < 2e-2, f"{site} differs alone vs batched ({rel:.2e}); padding leaks into the sums"


def token_lengths(rendered: list[Rendered], cfg: ExtractionRunConfig) -> list[int]:
    lengths = [len(r.input_ids) for r in rendered]
    assert max(lengths) <= cfg.emotion_vectors.extract.max_length, f"longest story is {max(lengths)} tokens"
    assert min(lengths) > cfg.stories.split_token, f"shortest story is {min(lengths)} tokens"
    print(f"[tokens] min {min(lengths)}, median {sorted(lengths)[len(lengths) // 2]}, max {max(lengths)}")
    return lengths


def extract_all(model, pad_id: int, cfg: ExtractionRunConfig, rendered: list[Rendered], order: list[int]):
    """Pooled sums for every story, batched in `order`. Returns `sums` per site and `counts` [story, 2]."""
    extract = cfg.emotion_vectors.extract
    sums: dict[str, torch.Tensor] = {}
    counts = torch.zeros(len(rendered), 2, dtype=torch.int64)
    for start in tqdm(range(0, len(order), extract.batch_size), desc="extract"):
        idx = order[start : start + extract.batch_size]
        out, c = batch_sums(model, extract.layer, [rendered[i] for i in idx], pad_id)
        for site, v in out.items():
            sums.setdefault(site, torch.zeros(len(rendered), v.shape[1], dtype=torch.float32))[idx] = v.cpu()
        counts[idx] = c.cpu()
    return sums, counts


def save_sums(cfg: ExtractionRunConfig, stories: list[Story], lengths, sums, counts, n_layers: int) -> None:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    save_file({**sums, "counts": counts}, str(cfg.run_dir / "sums.safetensors"))
    meta = {
        "config": json.loads(cfg.model_dump_json()),
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "n_layers": n_layers,
        "dims": {k: v.shape[1] for k, v in sums.items()},
        "stories": [s.model_dump(exclude={"text"}) | {"n_tokens": n} for s, n in zip(stories, lengths)],
    }
    (cfg.run_dir / "stories.json").write_text(json.dumps(meta, indent=1))
    print(f"[saved] {cfg.run_dir}/sums.safetensors, {len(sums)} arrays + counts")


def vectors_meta(cfg: ExtractionRunConfig, split: StorySplit, denoise: dict, extract_commit: str) -> dict:
    def count(mask, e):
        return int((mask & (split.label == e)).sum())

    return {
        "emotions": list(EMOTIONS),
        "sites": list(SITES),
        "keys": "<site>/<mean|neutral>/<raw|denoised> [12, dim]; <site>/neutral_pcs; <site>/grand_mean; <site>/neutral_mean",
        "average": f"mean over tokens {cfg.stories.split_token}+ per story, then over stories",
        "denoise": {site: info.model_dump() for site, info in denoise.items()},
        "n_train": {e: count(split.train, i) for i, e in enumerate(EMOTIONS)} | {"neutral": int((split.train & split.neutral).sum())},
        "n_heldout": {e: count(split.heldout, i) for i, e in enumerate(EMOTIONS)} | {"neutral": int((split.heldout & split.neutral).sum())},
        "model": cfg.model.name,
        "layer": cfg.emotion_vectors.extract.layer,
        "extract_git_commit": extract_commit,
    }  # fmt: skip


def site_checks(x, split: StorySplit, vectors: EmotionVectors, site: str, records: list[StoryRecord]) -> dict:
    """Denoised vectors for the labels; raw accuracy kept for comparison."""
    pcs = vectors.neutral_pcs(site).double()
    raw_correct = nearest_vector_scores(x, split).argmax(1) == split.label
    scores = nearest_vector_scores(x, split, pcs)
    top2 = scores.topk(2, dim=1)
    predicted = top2.indices[:, 0]
    correct = predicted == split.label
    emotional = split.emotional
    base = vectors.get(site, "mean", "raw").double()
    return {
        "acc_train_loo": correct[emotional & split.train].double().mean().item(),
        "acc_heldout": correct[emotional & split.heldout].double().mean().item(),
        "acc_train_loo_raw": raw_correct[emotional & split.train].double().mean().item(),
        "acc_heldout_raw": raw_correct[emotional & split.heldout].double().mean().item(),
        "confusion_all": confusion(predicted, split).tolist(),
        "split_half_cos": dict(zip(EMOTIONS, split_half_cosine(x, split))),
        "cos_mean_raw": cosine_matrix(base, base).tolist(),
        "cos_mean_denoised": cosine_matrix(remove(base, pcs), remove(base, pcs)).tolist(),
        "pred": {
            records[i].id: [EMOTIONS[top2.indices[i, 0]], EMOTIONS[top2.indices[i, 1]],
                            round(top2.values[i, 0].item() - scores[i, split.label[i]].item(), 4)]
            for i in emotional.nonzero().flatten().tolist()
        },
    }  # fmt: skip


def flagged_stories(checks: dict, records: list[StoryRecord], split: StorySplit) -> list[dict]:
    """Mislabelled at every site, sorted by the mean margin of the wrong vector over the story's own."""
    flagged = []
    for i in split.emotional.nonzero().flatten().tolist():
        r = records[i]
        preds = {s: checks["sites"][s]["pred"][r.id] for s in SITES}
        wrong = {s: p[0] for s, p in preds.items() if p[0] != r.emotion}
        if len(wrong) == len(SITES):
            margin = sum(p[2] for p in preds.values()) / len(SITES)
            flagged.append(
                {"id": r.id, "label": r.emotion, "heldout": r.heldout, "margin": round(margin, 4), "pred_by_site": wrong}
            )
    return sorted(flagged, key=lambda f: -f["margin"])


class EmotionVectorsMethod:
    def __init__(self, cfg: ExtractionRunConfig):
        assert isinstance(cfg.extraction, EmotionVectorsExtraction), "not an emotion_vectors run"
        self.cfg = cfg
        self.adapter = adapter_for(cfg.model)

    @cached_property
    def loaded(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Loaded by the first step that needs the model, then shared by the later steps."""
        return self.adapter.load(self.cfg.device)

    def capture(self) -> None:
        t0 = time.time()
        stories = load_stories(self.cfg.stories)
        examples_of = example_set(self.cfg.data)
        examples = examples_of.examples()
        assert [e.id for e in examples] == [s.id for s in stories], "the example set's order differs from the stories'"
        model, tokenizer = self.loaded
        n_layers = len(model.model.layers)
        layer = self.cfg.emotion_vectors.extract.layer
        print(
            f"[load] {self.cfg.model.name}: {n_layers} layers, hooking layer {layer}, {len(stories)} stories, {time.time() - t0:.0f}s"
        )

        rendered = [examples_of.render(e, tokenizer, self.adapter) for e in examples]
        lengths = token_lengths(rendered, self.cfg)
        order = sorted(range(len(stories)), key=lambda i: -lengths[i])  # similar lengths share a batch
        sums, counts = extract_all(model, tokenizer.pad_token_id, self.cfg, rendered, order)
        print(f"[extract] done, {time.time() - t0:.0f}s")

        padding_check(model, tokenizer.pad_token_id, self.cfg, stories, rendered, order, sums, counts)
        assert counts.sum() == sum(lengths), "counted tokens != tokenized tokens"
        assert all(torch.isfinite(v).all() for v in sums.values()), "non-finite sums"
        save_sums(self.cfg, stories, lengths, sums, counts, n_layers)

    def build(self) -> None:
        split = StorySplit.of(load_story_records(self.cfg.run_dir))
        vectors, denoise = RECIPES[self.cfg.emotion_vectors.vectors.recipe](
            late_means(self.cfg.run_dir), split, self.cfg.emotion_vectors.vectors
        )
        self.cfg.result_dir.mkdir(parents=True, exist_ok=True)
        vectors.save(self.cfg.result_dir / "vectors.safetensors")
        extract_commit = json.loads((self.cfg.run_dir / "stories.json").read_text())["git_commit"]
        (self.cfg.result_dir / "vectors_meta.json").write_text(
            json.dumps(vectors_meta(self.cfg, split, denoise, extract_commit), indent=1)
        )
        print(f"[saved] {self.cfg.result_dir}/vectors.safetensors ({len(vectors.tensors)} arrays), vectors_meta.json")

    def check(self) -> None:
        records = load_story_records(self.cfg.run_dir)
        split = StorySplit.of(records)
        means = late_means(self.cfg.run_dir)
        vectors = EmotionVectors.load(self.cfg.result_dir / "vectors.safetensors")
        denoise = json.loads((self.cfg.result_dir / "vectors_meta.json").read_text())["denoise"]
        checks = {"sites": {}}
        for site in SITES:
            checks["sites"][site] = c = site_checks(means[site], split, vectors, site, records)
            halves = sorted(c["split_half_cos"].values())
            print(f"[{site}] PCs {denoise[site]['k']} ({denoise[site]['variance_explained']:.2f} var) | denoised acc train-LOO "
                  f"{c['acc_train_loo']:.3f} heldout {c['acc_heldout']:.3f} (raw {c['acc_train_loo_raw']:.3f} / "
                  f"{c['acc_heldout_raw']:.3f}) | split-half cos min {halves[0]:.3f} median {halves[6]:.3f}")  # fmt: skip
        checks["flagged"] = flagged_stories(checks, records, split)
        checks["emotions"] = list(EMOTIONS)
        print(f"[flagged] {len(checks['flagged'])} stories mislabelled at all {len(SITES)} sites (denoised vectors)")
        (self.cfg.result_dir / "checks.json").write_text(json.dumps(checks))
        print(f"[saved] {self.cfg.result_dir}/checks.json")

    def select(self) -> None:
        """Nothing to select: the config fixes the layer, and every emotion keeps its vector."""

    def tokens(self) -> None:
        scores, values = story_token_scores(self.cfg, *self.loaded)
        save_token_scores(self.cfg.result_dir, scores, values)
        print(f"[saved] {self.cfg.result_dir}/token_scores.*: {len(scores.texts)} texts, {values.shape[0]} tokens")

    def report(self) -> None:
        path = write_report("emotion_vectors.html", build_emotion_vectors_report(self.cfg), self.cfg.result_dir / "report.html")
        print(f"[saved] {path}")
