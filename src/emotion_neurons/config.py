"""Typed run configs: a small shared base, and one config class per kind of experiment.

- `RunConfig` holds what every run has: run id, model, layer, device, folders.
- Each experiment's class adds only its own sections:
  - `VectorsRunConfig`: activations and emotion vectors (extract.py, build_vectors.py);
  - `PolesRunConfig`: neuron poles (poles.py);
  - `DirectionRunConfig`: a difference-of-means direction on instruction data (e.g. refusal), selected by its effect.
- Values come from two places, and the command line wins:
  - a YAML file given with `--config`, for the settings that distinguish one experiment from another, in git;
  - flags, for one-off changes on top of it, e.g. `--extract.batch-size 8`.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeVar

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


class RunConfig(BaseSettings):
    """What every run has. Experiments subclass it with their own sections."""

    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True)

    config: Path | None = Field(None, description="YAML file with defaults for any of the settings below")
    run_id: str = Field("dev", description="names runs/<run_id> (large, ignored) and results/<run_id> (committed)")
    model: str = "Qwen/Qwen3-8B"
    layer: int = 25
    device: str = "cuda"
    runs_dir: Path = Path("runs")
    results_dir: Path = Path("results")

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


class VectorsRunConfig(RunConfig):
    data: DataConfig = DataConfig()
    extract: ExtractConfig = ExtractConfig()
    vectors: VectorsConfig = VectorsConfig()


class PolesRunConfig(RunConfig):
    poles: PolesConfig = PolesConfig()


class InstructionSource(Section):
    """A JSON list at `url`, or a Hub dataset (`path`, `config`, `split`); instructions are read from `field`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str | None = None
    path: str | None = None
    config: str | None = None
    split: str = "train"
    field: str = "instruction"
    require_empty: str | None = Field(None, description="keep only rows whose field of this name is empty (e.g. Alpaca 'input')")


class InstructionSplits(Section):
    train: list[InstructionSource]
    val: list[InstructionSource]


class InstructionSetConfig(Section):
    """Instructions per category, sampled with a fixed seed (`datasets.instructions`)."""

    categories: dict[str, InstructionSplits]
    n_train: int = 128
    n_val: int = 32
    seed: int = 42


class DiffMeansConfig(Section):
    """A difference-of-means direction for one target category (`vectors.diff_means`)."""

    target: str
    baseline: Literal["all", "rest"] = Field(
        "rest", description="subtract the mean of all category means, or of the other categories' means"
    )
    pca_reference: str | None = Field(None, description="remove the top PCs of this category's examples (None: no PCA)")
    pca_variance: float = Field(0.5, gt=0, lt=1)


class DirectionSelectionConfig(Section):
    """The refusal paper's selection of a layer and post-instruction position (`refusal.select`)."""

    refusal_tokens: list[str] = Field(["I", "As"], description="first tokens of a refusal (the paper's for Qwen)")
    max_layer_fraction: float = Field(0.8, description="skip candidates in the last 20% of layers")
    max_kl: float = Field(0.1, description="ablating the direction may change harmless next-token predictions this much")
    batch_size: int = 32
    reuse_saved: bool = Field(
        False, description="reuse a saved direction.json/direction.safetensors (rerun the generation check only)"
    )


class GenerationCheckConfig(Section):
    max_new_tokens: int = 64
    batch_size: int = 16
    refusal_phrases: list[str] = [
        "I'm sorry", "I am sorry", "I apologize", "As an AI", "As a language model", "As an assistant", "I cannot",
        "I can't", "I am unable to", "I'm unable to", "I am not able to", "I'm not able to",
    ]  # fmt: skip


class DirectionRunConfig(RunConfig):
    """A difference-of-means direction on instruction data, selected and checked as in the refusal paper."""

    instructions: InstructionSetConfig
    direction: DiffMeansConfig
    selection: DirectionSelectionConfig = DirectionSelectionConfig()
    generation: GenerationCheckConfig = GenerationCheckConfig()


Config = TypeVar("Config", bound=RunConfig)


def load_run_config(config_class: type[Config], argv: Sequence[str]) -> Config:
    """The file is read first and handed over as init values, with the flags layered on top."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    known, _ = parser.parse_known_args(argv)
    values = yaml.safe_load(known.config.read_text()) if known.config else {}
    return config_class(_cli_parse_args=list(argv), **(values or {}))
