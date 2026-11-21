"""End-to-end modes report with checkpoints: harvest -> modes -> judge -> report.

- Each finished stage writes results/<run_id>/stages/<stage>.json: its outputs with their sizes, seconds, git commit,
  and the stage's summary.
- On a restart, a stage is skipped when its marker exists and every listed output is still there at the same size.
- When a stage runs, the markers of the stages after it are deleted: their inputs changed.
- With `pipeline.s3_prefix`, results/<run_id> is pulled at the start and pushed after every stage; runs/<run_id>
  (the 2-3 GB corpus pass) is pushed after the harvest and pulled only when the modes stage needs it.
  - So a new VM, or another machine, continues where the last one stopped.
- GPU stages (harvest, modes) stop the run cleanly on a machine without CUDA; rerun there with a GPU, or continue the
  CPU stages (judge, report) elsewhere from the pushed checkpoints.
- Inside a stage, the harvest reuses a saved corpus pass and the judge skips finished batch files.
"""

import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from .config import ModesRunConfig
from .harvest import run_harvest
from .judge import model_dir, run_judge
from .modes import run_modes
from .modes_report import run_report


@dataclass(frozen=True)
class Stage:
    name: str
    run: Callable[[ModesRunConfig], object]
    needs_gpu: bool
    outputs: Callable[[ModesRunConfig], list[Path]]


def harvest_outputs(cfg: ModesRunConfig) -> list[Path]:
    runs = [cfg.run_dir / f for f in ("corpus.safetensors", "hits.safetensors", "sources.json")]
    return [cfg.result_dir / "harvest.json", *runs]


def modes_outputs(cfg: ModesRunConfig) -> list[Path]:
    return [cfg.result_dir / "modes.json", *sorted((cfg.result_dir / "modes").glob("batch_*.txt"))]


def judge_outputs(cfg: ModesRunConfig) -> list[Path]:
    root = cfg.result_dir / "judgements"
    return [p for m in cfg.judge.models for p in sorted(model_dir(root, m).glob("batch_*.jsonl"))]


def report_outputs(cfg: ModesRunConfig) -> list[Path]:
    return [cfg.result_dir / "report.html", cfg.result_dir / "neurons.csv"]


STAGES = (
    Stage("harvest", run_harvest, needs_gpu=True, outputs=harvest_outputs),
    Stage("modes", run_modes, needs_gpu=True, outputs=modes_outputs),
    Stage("judge", run_judge, needs_gpu=False, outputs=judge_outputs),
    Stage("report", run_report, needs_gpu=False, outputs=report_outputs),
)


def marker_path(cfg: ModesRunConfig, stage: str) -> Path:
    return cfg.result_dir / "stages" / f"{stage}.json"


def is_done(cfg: ModesRunConfig, stage: Stage) -> bool:
    """The marker exists and its outputs are intact. With S3 checkpoints, the large runs/ files may be on S3 only:
    a machine that runs only judge and report never needs them, and modes pulls them when missing."""
    path = marker_path(cfg, stage.name)
    if not path.exists():
        return False
    outputs = json.loads(path.read_text())["outputs"]
    local = {p: size for p, size in outputs.items() if not (cfg.pipeline.s3_prefix and Path(p).is_relative_to(cfg.run_dir))}
    return bool(outputs) and all(Path(p).exists() and Path(p).stat().st_size == size for p, size in local.items())


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_marker(cfg: ModesRunConfig, stage: Stage, seconds: float, summary: object) -> None:
    outputs = {str(p): p.stat().st_size for p in stage.outputs(cfg)}
    assert outputs, f"stage {stage.name} finished without writing its outputs"
    marker = {
        "stage": stage.name, "finished": datetime.now(UTC).isoformat(timespec="seconds"), "seconds": round(seconds),
        "commit": git_commit(), "summary": summary, "outputs": outputs,
    }  # fmt: skip
    path = marker_path(cfg, stage.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(marker, indent=1, default=str))


def clear_later_markers(cfg: ModesRunConfig, stage: Stage) -> None:
    later = [s.name for s in STAGES[[s.name for s in STAGES].index(stage.name) + 1 :]]
    for name in later:
        marker_path(cfg, name).unlink(missing_ok=True)


def s3_sync(source: str, target: str) -> None:
    print(f"[sync] {source} -> {target}", flush=True)
    subprocess.run(["aws", "s3", "sync", source, target, "--only-show-errors"], check=True)


def remote(cfg: ModesRunConfig, part: str) -> str:
    return f"{cfg.pipeline.s3_prefix.rstrip('/')}/{cfg.run_id}/{part}"


def pull_results(cfg: ModesRunConfig) -> None:
    if cfg.pipeline.s3_prefix:
        cfg.result_dir.mkdir(parents=True, exist_ok=True)
        s3_sync(remote(cfg, "results"), str(cfg.result_dir))


def pull_runs_if_missing(cfg: ModesRunConfig) -> None:
    if cfg.pipeline.s3_prefix and not (cfg.run_dir / "corpus.safetensors").exists():
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        s3_sync(remote(cfg, "runs"), str(cfg.run_dir))


def push(cfg: ModesRunConfig, stage: Stage) -> None:
    if not cfg.pipeline.s3_prefix:
        return
    s3_sync(str(cfg.result_dir), remote(cfg, "results"))
    if stage.name == "harvest":
        s3_sync(str(cfg.run_dir), remote(cfg, "runs"))


def can_run_model(cfg: ModesRunConfig) -> bool:
    """A CUDA device that exists, or an explicit `device: cpu` (small test models)."""
    return cfg.device == "cpu" or (cfg.device.startswith("cuda") and torch.cuda.is_available())


def run_pipeline(cfg: ModesRunConfig) -> list[str]:
    """Runs every stage that is not done, in order, up to `pipeline.until`. Returns the stages it ran."""
    torch.set_grad_enabled(False)
    t_start = time.time()
    pull_results(cfg)
    ran = []
    for stage in STAGES:
        if is_done(cfg, stage):
            print(f"[pipeline] {stage.name}: done before, skipped", flush=True)
        elif stage.needs_gpu and not can_run_model(cfg):
            print(f"[pipeline] {stage.name} needs a GPU; stopping here. Rerun on a GPU machine to continue.", flush=True)
            return ran
        else:
            if stage.name == "modes":
                pull_runs_if_missing(cfg)
            print(f"[pipeline] {stage.name}: running", flush=True)
            clear_later_markers(cfg, stage)
            t0 = time.time()
            summary = stage.run(cfg)
            write_marker(cfg, stage, time.time() - t0, summary)
            push(cfg, stage)
            ran.append(stage.name)
            print(f"[pipeline] {stage.name}: finished in {time.time() - t0:.0f}s", flush=True)
        if stage.name == cfg.pipeline.until:
            break
    print(
        f"[pipeline] stopped after {cfg.pipeline.until or 'report'}; ran {ran or 'nothing'} | {time.time() - t_start:.0f}s",
        flush=True,
    )
    return ran
