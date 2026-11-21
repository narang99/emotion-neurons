"""Everything a run can be told, as one typed config.

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
    heldout_neutral_idx: list[int] = Field([10, 11], description="neutral stories with this story_idx (0-11) are held out")
    split_token: int = Field(50, description="vectors average from this token on (the paper's 50)")


class ExtractConfig(Section):
    batch_size: int = 16
    max_length: int = Field(512, description="a story longer than this fails the run, nothing is truncated")


class VectorsConfig(Section):
    recipe: Literal["mean_contrast"] = Field(
        "mean_contrast", description="how the vectors are built, see emotion_neurons.vectors.RECIPES"
    )
    denoise_variance: float = Field(
        0.5, gt=0, lt=1, description="project out the top neutral PCs that explain this share of variance"
    )


class CorpusConfig(Section):
    name: str
    path: str
    config: str | None = None
    split: str = "train"
    field: str = Field("text", description="a list field is read as chat messages {role, content}")
    tokens: int


class PolesConfig(Section):
    """The poles of a neuron: at or below its p3, or at or above its p97, over general text."""

    corpora: list[CorpusConfig] = [
        CorpusConfig(name="fineweb-edu", path="HuggingFaceFW/fineweb-edu", config="sample-10BT", tokens=70_000),
        CorpusConfig(name="tinystories", path="roneneldan/TinyStories", tokens=70_000),
        CorpusConfig(name="ultrachat", path="HuggingFaceH4/ultrachat_200k", split="train_sft", field="messages", tokens=70_000),
    ]
    quantiles: list[float] = [0.03, 0.97]
    window: int = 256
    min_window: int = 32
    skip_first: int = Field(4, description="the first tokens of each window are left out (attention-sink effects)")
    batch_size: int = 32
    seed: int = 0


class GreedySetsConfig(Section):
    """Greedy sets of one layer's neurons for each emotion's MLP write (`emotion_neurons.sparse`)."""

    sums_run: str = Field("stage2_l25", description="runs/<sums_run> holds the pooled sums and stories.json")
    rebuilt: float = Field(0.9, description="stop when the set rebuilds this share of the write")


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
    poles: PolesConfig = PolesConfig()
    greedy_sets: GreedySetsConfig = GreedySetsConfig()

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
