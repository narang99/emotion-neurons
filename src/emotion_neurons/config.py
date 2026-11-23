"""Typed run configs: a small shared base, and one config class per kind of experiment.

- `RunConfig` holds what every run has: run id, model, layer, device, folders.
- Each experiment's class adds only its own sections:
  - `VectorsRunConfig`: activations and emotion vectors (extract.py, build_vectors.py);
  - `PolesRunConfig`: neuron poles (poles.py);
  - `GreedySetsRunConfig`: greedy sets (greedy_sets.py);
  - `ModesRunConfig`: harvest, modes, judge, report and the end-to-end pipeline;
  - `ThemeRemovalRunConfig`: remove a theme's neurons from an emotion vector, held-out scores, token cosines;
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
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
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


class HarvestConfig(Section):
    """What chosen neurons fire on, at both poles (`emotion_neurons.harvest`)."""

    neurons_csv: Path = Field(Path("results/greedy_sets_l25/neurons/happy.csv"), description="a greedy-set CSV, in rank order")
    rebuilt_pct: float = Field(75.0, description="harvest the neurons up to the one that rebuilds this % of the write")
    n_controls: int = 20
    corpora: list[CorpusConfig] = [
        CorpusConfig(name="fineweb-edu", path="HuggingFaceFW/fineweb-edu", config="sample-10BT", tokens=700_000),
        CorpusConfig(name="tinystories", path="roneneldan/TinyStories", tokens=700_000),
        CorpusConfig(name="ultrachat", path="HuggingFaceH4/ultrachat_200k", split="train_sft", field="messages", tokens=600_000),
    ]
    high_quantiles: list[float] = [0.97, 0.99, 0.999, 0.9999]
    low_quantiles: list[float] = [0.03, 0.01, 0.001, 0.0001]
    examples_per_band: int = 8
    extreme_examples: int = 12
    hits_per_pole: int = Field(1000, description="hit positions kept per neuron and pole, for the mode clustering")
    all_neurons: bool = Field(False, description="harvest every neuron of the layer (`emotion_neurons.harvest_all`)")
    neuron_chunk: int = Field(1024, description="neurons per chunk in an all-neuron harvest")
    hits_per_band: int | None = Field(
        None, description="if set, sample up to this many hits from each band instead of `hits_per_pole` uniformly"
    )
    before: int = Field(60, description="tokens shown before the firing token")
    after: int = Field(8, description="tokens shown after it")
    window: int = 256
    min_window: int = 32
    skip_first: int = Field(4, description="the first tokens of each window are left out (attention-sink effects)")
    batch_size: int = 32
    seed: int = 0


class ModesConfig(Section):
    """Clusters of each harvested neuron's hits at each pole (`emotion_neurons.modes`); defaults of tag exp/stage3_l25."""

    hits_used: int = Field(500, description="hits clustered per neuron and pole (a subsample of the harvest's)")
    pca_dims: int = 50
    min_cluster_size: int = 10
    min_samples: int | None = 5
    assign_quantile: float = Field(0.1, description="a noise hit joins a cluster above this quantile of member cosines")
    exemplars: int = 5
    other_exemplars: int = 3
    neurons_per_batch: int = Field(8, description="at most this many neurons per text file for the judges")
    clusters_per_batch: int = Field(80, description="and at most this many clusters, so a judge's reply is not cut off")
    snippet_before: int = 30
    snippet_after: int = 8
    batch_size: int = 32
    checks: int = Field(200, description="hits whose recomputed activation is checked against the harvest")
    workers: int = Field(0, description="processes for the clustering in an all-neuron harvest; 0 means one per CPU")


class JudgeConfig(Section):
    """Blind judgements of mode clusters by cheap models on OpenRouter (`emotion_neurons.judge`)."""

    models: list[str] = ["anthropic/claude-haiku-5.5", "google/gemini-2.5-flash-lite"]
    key_file: Path = Field(Path("~/.config/openrouter/key"), description="read at run time, never printed or committed")
    workers: int = Field(8, description="requests in flight at once, per model")
    max_tokens: int = 8000
    scene: str | None = Field(
        None, description="the scene to count; default the run's concept. Another scene goes to judgements_<scene>/"
    )


class GreedySetsConfig(Section):
    """Greedy sets of one layer's neurons for each emotion's MLP write (`emotion_neurons.sparse`)."""

    sums_run: str = Field("stage2_l25", description="runs/<sums_run> holds the pooled sums and stories.json")
    rebuilt: float = Field(0.9, description="stop when the set rebuilds this share of the write")


class ThemesConfig(Section):
    """Themes of meaning across a run's cluster labels, by a strong model on OpenRouter (`emotion_neurons.themes`)."""

    model: str = "anthropic/claude-sonnet-5.5"
    n_min: int = 15
    n_max: int = 25
    batch_size: int = Field(300, description="clusters per assignment request")
    max_tokens: int = 16000


class PipelineConfig(Section):
    """Checkpoints of the end-to-end run (`emotion_neurons.pipeline`)."""

    s3_prefix: str | None = Field(
        None, description="if set, each stage's outputs are pushed to <s3_prefix>/<run_id>/ and pulled back on a restart"
    )
    until: Literal["harvest", "modes", "judge", "report"] | None = Field(None, description="stop after this stage")
    redo: list[Literal["harvest", "modes", "judge", "report"]] = Field(
        [], description="run these stages again even if done (their markers are cleared after the S3 pull)"
    )


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


class GreedySetsRunConfig(RunConfig):
    greedy_sets: GreedySetsConfig = GreedySetsConfig()


class ModesRunConfig(RunConfig):
    concept: str = Field(
        "happy", validation_alias=AliasChoices("concept", "emotion"),
        description="what the neuron set builds: an emotion, or another direction such as refusal (YAML key `emotion` also works)",
    )  # fmt: skip
    harvest: HarvestConfig = HarvestConfig()
    modes: ModesConfig = ModesConfig()
    judge: JudgeConfig = JudgeConfig()
    themes: ThemesConfig = ThemesConfig()
    pipeline: PipelineConfig = PipelineConfig()

    @model_validator(mode="after")
    def concept_matches_its_neurons(self) -> "ModesRunConfig":
        """A config that names one concept but harvests another's neuron set is a copy-paste slip; fail on it."""
        csv_concept = self.harvest.neurons_csv.stem
        assert csv_concept == self.concept, f"concept is {self.concept!r} but harvest.neurons_csv is {csv_concept}'s set"
        return self


class RemovalConfig(Section):
    """Which neurons to remove: those whose only emotion theme (on the emotion's side) is `theme`."""

    modes_run: str = Field(description="results/<modes_run> holds cluster_themes.csv and neurons.csv")
    theme: str
    not_emotion_themes: list[str] = Field(description="themes that do not count as the emotion (style, syntax, off-topic)")


class TokensConfig(Section):
    batch_size: int = 16
    stories: Literal["heldout", "all"] = Field("heldout", description="which stories get per-token scores")


class ThemeRemovalRunConfig(RunConfig):
    """Remove a theme's neurons from an emotion's write; held-out scores and per-token cosines (`theme_removal`)."""

    emotion: str
    sums_run: str = Field("stage2_l25", description="runs/<sums_run> holds the pooled sums and stories.json")
    data: DataConfig = DataConfig()
    removal: RemovalConfig
    tokens: TokensConfig = TokensConfig()


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


class DirectionNeuronsConfig(Section):
    """The neuron set behind a selected direction (`directions.run_direction_neurons`)."""

    concept: str = Field(description="names the neuron set (its CSV is neurons/<concept>.csv)")
    layer_offset: int = Field(-1, description="the MLP of the chosen layer plus this: -1 is the last MLP writing into it")
    rebuilt: float = Field(0.9, description="greedy set stops when it rebuilds this share of the MLP's write")
    batch_size: int = 32


class DirectionRunConfig(RunConfig):
    """A difference-of-means direction on instruction data, selected and checked as in the refusal paper."""

    instructions: InstructionSetConfig
    direction: DiffMeansConfig
    selection: DirectionSelectionConfig = DirectionSelectionConfig()
    generation: GenerationCheckConfig = GenerationCheckConfig()
    neurons: DirectionNeuronsConfig | None = None


Config = TypeVar("Config", bound=RunConfig)


def load_run_config(config_class: type[Config], argv: Sequence[str]) -> Config:
    """The file is read first and handed over as init values, with the flags layered on top."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    known, _ = parser.parse_known_args(argv)
    values = yaml.safe_load(known.config.read_text()) if known.config else {}
    return config_class(_cli_parse_args=list(argv), **(values or {}))
