"""Blind scene judgements and mode names for every cluster of a modes run, by cheap models on OpenRouter. CPU.

    python scripts/judge_modes.py --config configs/harvest_<emotion>_l25.yaml [--judge.models '["..."]']

- Reads results/<run_id>/modes/batch_XX.txt (scripts/modes.py).
- Judges scenes of the config's `emotion` (`emotion_neurons.judge`).
- Writes results/<run_id>/judgements/<model>/batch_XX.jsonl, one verdict per cluster.
  - A batch file whose output exists is skipped, so a stopped run resumes.
- Prints, per model, the clusters judged and missing, and the share judged an emotion scene (3+ of 5 windows);
  with two models, how often they agree.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from emotion_neurons.config import load_run_config
from emotion_neurons.judge import Verdict, cluster_keys, judge_batch, read_key

SCENE_AT = 3


def model_dir(root: Path, model: str) -> Path:
    return root / model.replace("/", "--")


def judge_file(model: str, emotion: str, batch: Path, out: Path, key: str, max_tokens: int) -> tuple[str, int, int]:
    verdicts, missing = judge_batch(model, emotion, batch.read_text(), key, max_tokens)
    out.write_text("".join(v.model_dump_json() + "\n" for v in verdicts))
    return batch.name, len(verdicts), len(missing)


def run_model(model: str, batches: list[Path], root: Path, key: str, cfg) -> None:
    folder = model_dir(root, model)
    folder.mkdir(parents=True, exist_ok=True)
    todo = [b for b in batches if not (folder / b.with_suffix(".jsonl").name).exists()]
    print(f"[{model}] {len(batches) - len(todo)} batch files done before, {len(todo)} to go", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(cfg.judge.workers) as pool:
        futures = [
            pool.submit(judge_file, model, cfg.emotion, b, folder / b.with_suffix(".jsonl").name, key, cfg.judge.max_tokens)
            for b in todo
        ]
        for done, future in enumerate(as_completed(futures), 1):
            name, judged, missing = future.result()
            note = f", {missing} missing" if missing else ""
            print(f"[{model}] {done}/{len(todo)} {name}: {judged} clusters{note}, {time.time() - t0:.0f}s", flush=True)


def load_verdicts(folder: Path) -> dict[tuple, Verdict]:
    verdicts = {}
    for path in sorted(folder.glob("batch_*.jsonl")):
        for line in path.read_text().splitlines():
            v = Verdict.model_validate_json(line)
            verdicts[(v.neuron, v.pole, v.cluster)] = v
    return verdicts


def report(models: list[str], root: Path, expected: set[tuple], emotion: str) -> None:
    in_scene = {}
    for model in models:
        v = load_verdicts(model_dir(root, model))
        in_scene[model] = {k: x.scene_windows >= SCENE_AT for k, x in v.items()}
        share = sum(in_scene[model].values()) / max(1, len(v))
        print(f"[{model}] {len(v)} of {len(expected)} clusters judged, {len(expected - v.keys())} missing; "
              f"{100 * share:.0f}% judged {emotion} scenes")  # fmt: skip
    if len(models) == 2:
        a, b = (in_scene[m] for m in models)
        both = a.keys() & b.keys()
        agree = sum(a[k] == b[k] for k in both) / max(1, len(both))
        print(f"[agreement] the two models agree on {emotion} vs not for {100 * agree:.0f}% of {len(both)} clusters")


def main():
    cfg = load_run_config(sys.argv[1:])
    batches = sorted((cfg.result_dir / "modes").glob("batch_*.txt"))
    expected = {k for b in batches for k in cluster_keys(b.read_text())}
    root = cfg.result_dir / "judgements"
    key = read_key(cfg.judge.key_file)
    print(
        f"[input] {len(batches)} batch files, {len(expected)} clusters, emotion {cfg.emotion}, models {cfg.judge.models}",
        flush=True,
    )
    for model in cfg.judge.models:
        run_model(model, batches, root, key, cfg)
    report(cfg.judge.models, root, expected, cfg.emotion)
    print(f"[saved] {root}")


if __name__ == "__main__":
    main()
