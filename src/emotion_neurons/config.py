"""Typed run configs: a small shared base, and one config class per kind of experiment.

- `RunConfig` holds what every run has: run id, model, device, folders.
  - `model` is a union on `family`: `hf` (native transformers) or `qwen1` (converted, at a pinned revision).
- Each experiment's class adds only its own sections:
  - `ExtractionRunConfig`: a data source and an extraction method (extract.py), two unions on `kind`:
    - data: `stories` or `instructions`, both read through `datasets.ExampleSet`;
    - extraction: `emotion_vectors` (the emotions paper) or `direction` (the refusal paper's selection and checks);
  - `PolesRunConfig`: neuron poles (poles.py).
- Values come from two places, and the command line wins:
  - a YAML file given with `--config`, for the settings that distinguish one experiment from another, in git;
  - flags, for one-off changes on top of it, e.g. `--extract.batch-size 8`.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, TypeVar

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from .emotions import LABELS, Site

Baseline = Literal["mean", "neutral"]
Variant = Literal["raw", "denoised"]


class Section(BaseModel):
    """A typo in a config file should fail, not be ignored."""

    model_config = ConfigDict(extra="forbid")


class HFModel(Section):
    """A native transformers model, loaded with its own chat template (`models.adapters`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: Literal["hf"] = "hf"
    name: str = "Qwen/Qwen3-8B"
    revision: str | None = None


class Qwen1Model(Section):
    """A Qwen-1 chat model, converted to Qwen2 (`models.qwen1`).

    - The revision is required: the repo's remote tokenizer code is downloaded and run, so it must not change.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: Literal["qwen1"] = "qwen1"
    name: str = "Qwen/Qwen-7B-Chat"
    revision: str


def hf_shorthand(value: object) -> object:
    """`model: Qwen/Qwen3-8B` means `{family: hf, name: Qwen/Qwen3-8B}`."""
    return {"family": "hf", "name": value} if isinstance(value, str) else value


ModelConfig = Annotated[HFModel | Qwen1Model, Field(discriminator="family"), BeforeValidator(hf_shorthand)]


class StoriesData(Section):
    """The emotion stories, plain text, read from token `split_token` on (`datasets.stories`)."""

    kind: Literal["stories"] = "stories"
    stories_dir: Path = Path("data/stories")
    heldout_story_idx: list[int] = Field(
        [5], description="emotional stories with this story_idx (0-5) are held out of the vectors"
    )
    heldout_neutral_idx: list[int] = Field([10, 11], description="neutral stories with this story_idx (0-11) are held out")
    split_token: int = Field(50, description="vectors average from this token on (the paper's 50)")

    @property
    def categories(self) -> list[str]:
        return list(LABELS)


class ExtractConfig(Section):
    layer: int = 25
    batch_size: int = 16
    max_length: int = Field(512, description="a story longer than this fails the run, nothing is truncated")


class VectorsConfig(Section):
    recipe: Literal["mean_contrast"] = Field(
        "mean_contrast", description="how the vectors are built, see emotion_neurons.vectors.RECIPES"
    )
    denoise_variance: float = Field(
        0.5, gt=0, lt=1, description="project out the top neutral PCs that explain this share of variance"
    )


class StoryTokensConfig(Section):
    """Per-token scores of held-out stories against the emotion vectors (`token_scores`)."""

    site: Site = Field("resid_mid", description="the site whose denoised vectors score each token")
    batch_size: int = 16
    stories_per_label: int = Field(2, description="token views per emotion (and neutral) in the report")


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
    model: ModelConfig = HFModel()
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


class PolesRunConfig(RunConfig):
    layer: int = 25
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


class InstructionsData(Section):
    """Instructions per category, sampled with a fixed seed, in the model's chat format (`datasets.instructions`)."""

    kind: Literal["instructions"] = "instructions"
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


class GenerationCheckConfig(Section):
    max_new_tokens: int = 64
    batch_size: int = 16
    refusal_phrases: list[str] = [
        "I'm sorry", "I am sorry", "I apologize", "As an AI", "As a language model", "As an assistant", "I cannot",
        "I can't", "I am unable to", "I'm unable to", "I am not able to", "I'm not able to",
    ]  # fmt: skip


class DirectionTokensConfig(Section):
    """Per-token projections of val prompts and their saved plain replies onto the direction (`token_scores`)."""

    batch_size: int = 16
    prompts_per_category: int = Field(6, description="token views per category in the report")


DataConfig = Annotated[StoriesData | InstructionsData, Field(discriminator="kind")]


class EmotionVectorsExtraction(Section):
    """The emotions paper: pooled means at one layer, then a vector recipe, checked on held-out stories."""

    kind: Literal["emotion_vectors"] = "emotion_vectors"
    extract: ExtractConfig = ExtractConfig()
    vectors: VectorsConfig = VectorsConfig()
    tokens: StoryTokensConfig = StoryTokensConfig()


class DirectionExtraction(Section):
    """The refusal paper: means at every layer and post-instruction position, a candidate chosen by its effect,
    then a generation check."""

    kind: Literal["direction"] = "direction"
    diff_means: DiffMeansConfig
    selection: DirectionSelectionConfig = DirectionSelectionConfig()
    generation: GenerationCheckConfig = GenerationCheckConfig()
    tokens: DirectionTokensConfig = DirectionTokensConfig()


Extraction = Annotated[EmotionVectorsExtraction | DirectionExtraction, Field(discriminator="kind")]

Step = Literal["capture", "build", "select", "check"]


class ExtractionRunConfig(RunConfig):
    """A data source and an extraction method. The method's steps run in order (`emotion_neurons.extraction`)."""

    data: DataConfig
    extraction: Extraction
    steps: list[Step] | None = Field(None, description="run only these steps (None: all of the method's steps)")

    @model_validator(mode="after")
    def method_fits_data(self) -> "ExtractionRunConfig":
        match self.extraction:
            case EmotionVectorsExtraction():
                assert isinstance(self.data, StoriesData), "emotion_vectors needs stories: the checks use story topics"
            case DirectionExtraction(diff_means=d):
                others = [c for c in self.data.categories if c != d.target]
                assert d.target in self.data.categories, f"target {d.target!r} is not a data category"
                assert len(others) == 1, f"direction selection needs one category besides the target, got {others}"
        return self

    @property
    def emotion_vectors(self) -> EmotionVectorsExtraction:
        assert isinstance(self.extraction, EmotionVectorsExtraction), "not an emotion_vectors run"
        return self.extraction

    @property
    def direction(self) -> DirectionExtraction:
        assert isinstance(self.extraction, DirectionExtraction), "not a direction run"
        return self.extraction

    @property
    def stories(self) -> StoriesData:
        assert isinstance(self.data, StoriesData), "not a stories run"
        return self.data

    @property
    def instructions(self) -> InstructionsData:
        assert isinstance(self.data, InstructionsData), "not an instructions run"
        return self.data


Config = TypeVar("Config", bound=RunConfig)


def load_run_config(config_class: type[Config], argv: Sequence[str]) -> Config:
    """The file is read first and handed over as init values, with the flags layered on top."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    known, _ = parser.parse_known_args(argv)
    values = yaml.safe_load(known.config.read_text()) if known.config else {}
    return config_class(_cli_parse_args=list(argv), **(values or {}))
