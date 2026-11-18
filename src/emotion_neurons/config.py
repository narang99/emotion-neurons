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


class NeuronSetConfig(Section):
    vectors_run: str = Field("stage2_l25", description="results/<vectors_run>/vectors.safetensors holds the write and dA")
    dim_share: float = Field(0.9, gt=0, le=1, description="share of delta's positive (and negative) mass the kept dims cover")
    neuron_share: float = Field(
        0.9, gt=0, le=1, description="share of a dim's positive (and negative) terms the kept neurons cover"
    )


class CorpusConfig(Section):
    name: str
    path: str
    config: str | None = None
    split: str = "train"
    field: str = Field("text", description="a list field is read as chat messages {role, content}")
    tokens: int


class HarvestConfig(Section):
    corpora: list[CorpusConfig] = [
        CorpusConfig(name="fineweb-edu", path="HuggingFaceFW/fineweb-edu", config="sample-10BT", tokens=700_000),
        CorpusConfig(name="tinystories", path="roneneldan/TinyStories", tokens=650_000),
        CorpusConfig(name="ultrachat", path="HuggingFaceH4/ultrachat_200k", split="train_sft", field="messages", tokens=650_000),
    ]
    window: int = 256
    min_window: int = 32
    skip_first: int = Field(4, description="the first tokens of each window are left out of percentiles and hits")
    batch_size: int = 32
    n_controls: int = 50
    max_neurons: int = Field(1000, description="harvest the whole Stage 3 set up to this size, else the ranked top")
    per_group: int = Field(25, description="when ranking: neurons per (emotion, pole), see emotion_neurons.selection")
    seed: int = 0
    percentiles: list[float] = [0.97, 0.99, 0.999, 0.9999]
    percentile_chunk: int = 64
    examples_per_band: int = 20
    top_k: int = 50
    hits_per_neuron: int = Field(512, description="p97+ hits sampled per neuron for mode clustering (plus the top_k)")
    neurons_file: Path | None = Field(None, description="selection.json from min_set.py; overrides the Stage 3 ranking")


class ModesConfig(Section):
    per_group: int = Field(10, description="neurons per (emotion, neuron pole), by summed term over the kept dims")
    n_controls: int = 20
    pca_dims: int = 50
    min_cluster_size: int = 10
    min_samples: int | None = 5
    assign_quantile: float = Field(0.1, description="a noise hit joins a cluster above this quantile of member cosines")
    exemplars: int = 5
    noise_exemplars: int = 3
    neurons_per_batch: int = 10


class MinSetConfig(Section):
    greedy_steps: int = Field(2000, description="greedy forward-selection steps per emotion (stops at cos 0.99)")
    select: int = Field(64, description="neurons for the mode analysis, round-robin down the greedy orders")


class SparseSetConfig(Section):
    taus: list[float] = Field([0.8, 0.9, 0.95], description="target cosines for the smallest-set search")
    sizes: list[int] = Field([16, 64, 128, 256, 512, 1024, 2048], description="set sizes reported")
    greedy_steps: int = 2500
    max_swaps: int = Field(2000, description="1-swaps per polish")
    emotions: list[str] = Field([], description="restrict to these emotions (empty: all 12)")
    target: Literal["raw", "denoised"] = Field("raw", description="denoised: neutral PCs of the write removed")
    seed: int = Field(0, description="split of the train stories into halves for cross-fit greedy")
    shrink_kappas: list[float] = Field([1.0, 4.0, 9.0], description="shrunk-target greedy: r = max(0, 1 - kappa/t^2)")


class SparseLayersConfig(Section):
    source_run: str = Field("layers_all", description="all-layer pooled sums from extract_layers.py")
    layers: list[int] = Field([], description="layers to run (empty: all)")
    greedy_steps: int = 2500
    target_cos: float = Field(0.9, description="also report the smallest greedy prefix reaching this train cosine")
    seed: int = 0


class CircuitsConfig(Section):
    vectors_file: Path = Field(Path("results/layers_all/layer_vectors.safetensors"), description="from layer_vectors.py")
    sparse_layers_file: Path = Field(Path("results/sparse_layers/sparse_layers.json"), description="emotion sets per layer")
    target_layer: int = Field(25, description="the emotion score is read from this layer's resid_post")
    stories_per_emotion: int = Field(5, description="held-out stories per emotion")
    sizes: list[int] = Field([100, 300, 1000, 3000, 10000], description="circuit sizes checked by mean ablation")
    report_size: int = Field(1000, description="circuit size whose nodes are split by emotion-set membership")
    seed: int = 0


class ProbeConfig(Section):
    stories_dir: Path = Path("data/adult_stories")
    source_run: str = Field("top64_l25", description="harvest + modes run whose neurons, p97 and modes are probed")
    skip_first: int = 4
    context_before: int = 20
    context_after: int = 4
    examples_per_neuron: int = 8


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
    neuron_set: NeuronSetConfig = NeuronSetConfig()
    harvest: HarvestConfig = HarvestConfig()
    modes: ModesConfig = ModesConfig()
    min_set: MinSetConfig = MinSetConfig()
    sparse_set: SparseSetConfig = SparseSetConfig()
    sparse_layers: SparseLayersConfig = SparseLayersConfig()
    circuits: CircuitsConfig = CircuitsConfig()
    probe: ProbeConfig = ProbeConfig()

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
