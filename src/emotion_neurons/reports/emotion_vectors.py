"""Report data for one run of emotion vectors (scripts/build_vectors.py): read only from its saved results.

- Reads results/<run_id>/vectors_meta.json and checks.json. No model and no story sums are needed.
- Accuracy: nearest-vector labels of each story, 12-way, train stories left out of their own emotion's mean.
- Confusion: all emotional stories (train left out, and held out), denoised vectors.
"""

import json
from pathlib import Path

from pydantic import BaseModel

from ..config import VectorsRunConfig
from .heatmap import Heatmap, heatmap


class SiteAccuracy(BaseModel):
    site: str
    train_raw: float
    train_denoised: float
    heldout_raw: float
    heldout_denoised: float
    pcs_removed: int
    variance_removed: float


class SplitHalfRow(BaseModel):
    emotion: str
    cosines: list[float]


class FlaggedStory(BaseModel):
    id: str
    label: str
    heldout: bool
    margin: float
    predicted: list[str]


class SiteMaps(BaseModel):
    site: str
    confusion: Heatmap
    cosine_raw: Heatmap
    cosine_denoised: Heatmap


class EmotionVectorsReport(BaseModel):
    title: str
    model: str
    layer: int
    run_id: str
    average: str
    emotions: list[str]
    sites: list[str]
    accuracy: list[SiteAccuracy]
    chance: float
    maps: list[SiteMaps]
    split_half: list[SplitHalfRow]
    flagged: list[FlaggedStory]


def site_accuracy(site: str, checks: dict, denoise: dict) -> SiteAccuracy:
    c = checks["sites"][site]
    return SiteAccuracy(
        site=site, train_raw=c["acc_train_loo_raw"], train_denoised=c["acc_train_loo"],
        heldout_raw=c["acc_heldout_raw"], heldout_denoised=c["acc_heldout"],
        pcs_removed=denoise[site]["k"], variance_removed=denoise[site]["variance_explained"],
    )  # fmt: skip


def off_diagonal(matrix: list[list[float]]) -> list[list[float | None]]:
    return [[None if i == j else v for j, v in enumerate(row)] for i, row in enumerate(matrix)]


def site_maps(site: str, checks: dict, emotions: list[str]) -> SiteMaps:
    c = checks["sites"][site]
    cosine = "Cosine between emotion vectors ({}). The diagonal (1) is left blank, so it does not set the scale."
    return SiteMaps(
        site=site,
        confusion=heatmap(
            f"{site}: confusion", c["confusion_all"], emotions, emotions, scale="sequential", fmt="d",
            caption="Stories per true emotion (row) and nearest vector (column). Each row holds 120 stories.",
            row_title="true", col_title="nearest vector",
        ),
        cosine_raw=heatmap(f"{site}: cosine, raw", off_diagonal(c["cos_mean_raw"]), emotions, emotions, scale="diverging",
                           caption=cosine.format("raw"), row_title="emotion", col_title="emotion"),
        cosine_denoised=heatmap(f"{site}: cosine, denoised", off_diagonal(c["cos_mean_denoised"]), emotions, emotions,
                                scale="diverging", caption=cosine.format("neutral PCs removed"), row_title="emotion", col_title="emotion"),
    )  # fmt: skip


def flagged_story(f: dict, sites: list[str]) -> FlaggedStory:
    return FlaggedStory(id=f["id"], label=f["label"], heldout=f["heldout"], margin=f["margin"],
                        predicted=[f["pred_by_site"][s] for s in sites])  # fmt: skip


def build_emotion_vectors_report(cfg: VectorsRunConfig) -> EmotionVectorsReport:
    meta = json.loads(Path(cfg.result_dir / "vectors_meta.json").read_text())
    checks = json.loads(Path(cfg.result_dir / "checks.json").read_text())
    emotions, sites = meta["emotions"], meta["sites"]
    split_half = [SplitHalfRow(emotion=e, cosines=[checks["sites"][s]["split_half_cos"][e] for s in sites]) for e in emotions]
    return EmotionVectorsReport(
        title=f"Emotion vectors: {cfg.model}, layer {cfg.layer}",
        model=cfg.model, layer=cfg.layer, run_id=cfg.run_id, average=meta["average"],
        emotions=emotions, sites=sites,
        accuracy=[site_accuracy(s, checks, meta["denoise"]) for s in sites], chance=1 / len(emotions),
        maps=[site_maps(s, checks, emotions) for s in sites],
        split_half=split_half,
        flagged=[flagged_story(f, sites) for f in checks["flagged"]],
    )  # fmt: skip
