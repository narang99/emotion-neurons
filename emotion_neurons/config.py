"""Everything a Stage 2 run can be told, as one typed config.

Values come from two places, and the command line wins:

- a YAML file given with `--config`, for the settings that distinguish one experiment from
  another, so they live in git
- flags, for one-off changes on top of it, e.g. `--extract.batch-size 8`

Every script of a run reads the same file, so one commit reproduces the whole run.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

EMOTIONS = (
    "happy", "inspired", "loving", "proud", "calm", "desperate",
    "angry", "guilty", "sad", "afraid", "nervous", "surprised",
)  # fmt: skip

HOOKED_SITES = ("resid_mid", "resid_post", "mlp_act")
"""Recorded with hooks. resid_mid: input to post_attention_layernorm. resid_post: the layer's output.
mlp_act: input to down_proj, silu(gate) * up."""

SITES = (*HOOKED_SITES, "mlp_out")
"""mlp_out = resid_post - resid_mid, derived. Exact for pooled sums too, since pooling is linear."""

RANGES = ("early", "late")
"""early: tokens [0, split_token). late: tokens [split_token, end)."""

Baseline = Literal["mean", "neutral"]
Variant = Literal["raw", "denoised"]


class Section(BaseModel):
    """A typo in a config file should fail, not be ignored."""

    model_config = ConfigDict(extra="forbid")


class DataConfig(Section):
    stories_dir: Path = Path("data/stories")
    heldout_story_idx: list[int] = Field(
        [5], description="emotional stories with this story_idx (0-5) are held out of the vectors"
    )
    heldout_neutral_idx: list[int] = Field(
        [10, 11], description="neutral stories with this story_idx (0-11) are held out"
    )
    split_token: int = Field(50, description="vectors average from this token on (the paper's 50)")


class ExtractConfig(Section):
    batch_size: int = 16
    max_length: int = Field(512, description="a story longer than this fails the run, nothing is truncated")


class VectorsConfig(Section):
    denoise_variance: float = Field(
        0.5, gt=0, lt=1, description="project out the top neutral PCs that explain this share of variance"
    )


class ProjectConfig(Section):
    vectors: list[str] = Field(
        ["mean/raw", "mean/denoised"],
        description="baseline/variant pairs whose unit vectors every held-out token is projected onto",
    )
    batch_size: int = 16
    logit_lens_k: int = 20


class ReportConfig(Section):
    story_site: Literal["resid_mid", "resid_post", "mlp_act", "mlp_out"] = Field(
        "resid_post", description="site whose projections colour the per-story token pages"
    )
    story_vector: str = "mean/denoised"


class RunConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True)

    config: Path | None = Field(None, description="YAML file with defaults for any of the settings below")
    run_id: str = Field("dev", description="names runs/<run_id> (large, ignored) and results/<run_id> (committed)")
    model: str = "Qwen/Qwen3-8B"
    layer: int = 25
    device: str = "cuda"
    runs_dir: Path = Path("runs")
    results_dir: Path = Path("results")
    data: DataConfig = DataConfig()
    extract: ExtractConfig = ExtractConfig()
    vectors: VectorsConfig = VectorsConfig()
    project: ProjectConfig = ProjectConfig()
    report: ReportConfig = ReportConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """An environment variable named `MODEL` or `DEVICE` must not silently override a field."""
        return (init_settings,)

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.run_id

    @property
    def result_dir(self) -> Path:
        return self.results_dir / self.run_id


def load_run_config(argv: Sequence[str]) -> RunConfig:
    """The file is read first and handed over as init values, with the flags layered on top."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    known, _ = parser.parse_known_args(argv)
    values = yaml.safe_load(known.config.read_text()) if known.config else {}
    return RunConfig(_cli_parse_args=list(argv), **(values or {}))
