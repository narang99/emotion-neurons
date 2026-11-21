"""Blind judgements of mode clusters by a cheap model on OpenRouter.

- Input: a modes batch file (`scripts/modes.py`): per neuron and pole, clusters of exemplar windows with the
  firing token marked [[like this]]. The judge sees no labels, no emotion and no reason the neuron was chosen.
- Per cluster the judge returns: a short mode name, how many windows sit in a scene of the emotion, and the scenes.
- Scene, not word: for happy, a marked "and" inside a joyful scene counts; the word "happy" in a sad scene does not.
- The API key is read from a file at run time and never printed.
- The judge reports the count under an emotion-named key ("happy_windows", "angry_windows"); it is stored as
  `scene_windows`. A generic key name ("scene_windows") made Gemini 2.5 Flash-Lite write "windows" twice instead.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel

from .config import ModesRunConfig

API = "https://openrouter.ai/api/v1/chat/completions"

SCENES = {
    "happy": ("happy or joyful", "laughing, celebrating, having fun, delight, warm happy moments", '"happy"', "sad or neutral"),
    "angry": (
        "angry",
        "arguing, shouting, rage, frustration, resentment, someone furious or fed up",
        '"angry"',
        "calm or neutral",
    ),
    "sad": ("sad", "grief, loss, crying, loneliness, disappointment", '"sad"', "happy or neutral"),
    "afraid": ("frightening or fearful", "danger, threat, dread, panic, someone scared", '"afraid"', "safe or neutral"),
}
"""(scene words, what such scenes look like, the emotion word, the scenes where that word does not count)."""


def scene_hint(emotion: str) -> tuple[str, str, str, str]:
    return SCENES.get(emotion, (emotion, f"people feeling {emotion}", f'"{emotion}"', "neutral"))


def instructions(emotion: str) -> str:
    scene, looks, word, elsewhere = scene_hint(emotion)
    return f"""You read clusters of text windows. Each window has one token marked [[like this]]; a number in brackets before it is a neuron's activation there, which you can ignore.
The file is organised as: "##### neuron N", then "=== high pole" or "=== low pole", then "--- cluster K" followed by its windows. Ignore "--- unclustered (sample)" groups.

For every cluster, return:
- "mode": what the marked tokens and their contexts have in common, in at most 15 words;
- "{emotion}_windows": how many of its windows sit in a {scene} SCENE. Judge the situation and mood of the surrounding text ({looks}). Do not judge by the marked word alone: a marked "and", "so", "." or "was" inside such a scene counts; the word {word} inside a {elsewhere} scene does not. Recipes, technical text, news and neutral narration do not count;
- "windows": how many windows the cluster has;
- "scene": what the scenes are, in 3 to 8 words.

Answer with one JSON object per line and nothing else, in file order, one line per cluster:
{{"neuron": N, "pole": "high" or "low", "cluster": K, "mode": "...", "{emotion}_windows": 0, "windows": 5, "scene": "..."}}"""


class Verdict(BaseModel):
    neuron: int
    pole: str
    cluster: int
    mode: str
    scene_windows: int
    windows: int
    scene: str


def cluster_keys(text: str) -> list[tuple[int, str, int]]:
    """(neuron, pole, cluster) of every cluster in a batch file, in order."""
    keys, neuron, pole = [], None, None
    for line in text.splitlines():
        if m := re.match(r"##### neuron (\d+)", line):
            neuron = int(m.group(1))
        elif m := re.match(r"=== (high|low) pole", line):
            pole = m.group(1)
        elif m := re.match(r"--- cluster (\d+)", line):
            keys.append((neuron, pole, int(m.group(1))))
    return keys


def parse_verdicts(content: str) -> list[Verdict]:
    """Every JSON object in the reply that reads as a verdict; anything else is skipped."""
    verdicts = []
    for match in re.finditer(r"\{[^{}]*\}", content):
        try:
            verdicts.append(Verdict(**with_scene_count(json.loads(match.group(0)))))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return verdicts


def read_verdict(line: str) -> Verdict:
    """One stored verdict; older files keep the emotion-named count ("happy_windows")."""
    return Verdict(**with_scene_count(json.loads(line)))


def with_scene_count(row: dict) -> dict:
    """Move the emotion-named count ("happy_windows", "angry_windows", ...) to `scene_windows`."""
    if "scene_windows" not in row:
        named = [k for k in row if k.endswith("_windows")]
        if named:
            row["scene_windows"] = row.pop(named[0])
    return row


def ask(model: str, system: str, user: str, key: str, max_tokens: int, attempts: int = 4) -> str:
    """One chat completion; retries with backoff on network errors, rate limits and server errors."""
    body = json.dumps({
        "model": model, "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }).encode()  # fmt: skip
    request = urllib.request.Request(
        API, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                reply = json.loads(response.read())
            return reply["choices"][0]["message"]["content"] or ""
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
            status = getattr(error, "code", None)
            if status in (400, 401, 402, 403) or attempt == attempts - 1:
                raise RuntimeError(f"{model}: request failed ({status or type(error).__name__})") from None
            time.sleep(5 * 2**attempt)
    raise AssertionError("unreachable")


def judge_batch(
    model: str, emotion: str, text: str, key: str, max_tokens: int
) -> tuple[list[Verdict], list[tuple[int, str, int]]]:
    """Verdicts for every cluster of one batch file. Clusters missing from the first reply are asked again once."""
    system = instructions(emotion)
    expected = cluster_keys(text)
    found = {(v.neuron, v.pole, v.cluster): v for v in parse_verdicts(ask(model, system, text, key, max_tokens))}
    missing = [k for k in expected if k not in found]
    if missing:
        wanted = "\n".join(f"neuron {n}, {p} pole, cluster {c}" for n, p, c in missing)
        retry = f"{text}\n\nReturn lines only for these clusters, which were missing:\n{wanted}"
        for v in parse_verdicts(ask(model, system, retry, key, max_tokens)):
            found.setdefault((v.neuron, v.pole, v.cluster), v)
    verdicts = [found[k] for k in expected if k in found]
    return verdicts, [k for k in expected if k not in found]


def read_key(path: Path) -> str:
    """OPENROUTER_API_KEY if set (e.g. from a Colab secret), else the key file. Never printed."""
    key = os.environ.get("OPENROUTER_API_KEY") or path.expanduser().read_text().strip()
    assert key.startswith("sk-or-"), f"neither OPENROUTER_API_KEY nor {path} holds an OpenRouter key"
    return key


SCENE_AT = 3
"""A cluster is a scene of the emotion for a judge when 3+ of its 5 windows are."""


def model_dir(root: Path, model: str) -> Path:
    return root / model.replace("/", "--")


def judge_file(model: str, emotion: str, batch: Path, out: Path, key: str, max_tokens: int) -> tuple[str, int, int]:
    verdicts, missing = judge_batch(model, emotion, batch.read_text(), key, max_tokens)
    out.write_text("".join(v.model_dump_json() + "\n" for v in verdicts))
    return batch.name, len(verdicts), len(missing)


def judge_model(model: str, emotion: str, batches: list[Path], root: Path, key: str, workers: int, max_tokens: int) -> None:
    """Every batch file without an output yet; finished files are skipped, so a stopped run resumes."""
    folder = model_dir(root, model)
    folder.mkdir(parents=True, exist_ok=True)
    todo = [b for b in batches if not (folder / b.with_suffix(".jsonl").name).exists()]
    print(f"[{model}] {len(batches) - len(todo)} batch files done before, {len(todo)} to go", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        futures = [
            pool.submit(judge_file, model, emotion, b, folder / b.with_suffix(".jsonl").name, key, max_tokens) for b in todo
        ]
        for done, future in enumerate(as_completed(futures), 1):
            name, judged, missing = future.result()
            note = f", {missing} missing" if missing else ""
            print(f"[{model}] {done}/{len(todo)} {name}: {judged} clusters{note}, {time.time() - t0:.0f}s", flush=True)


def load_verdicts(folder: Path) -> dict[tuple, Verdict]:
    verdicts = {}
    for path in sorted(folder.glob("batch_*.jsonl")):
        for line in path.read_text().splitlines():
            v = read_verdict(line)
            verdicts[(v.neuron, v.pole, v.cluster)] = v
    return verdicts


def judge_summary(models: list[str], root: Path, expected: set[tuple], emotion: str) -> dict:
    """Per model: clusters judged and missing, share judged an emotion scene; with two models, their agreement."""
    summary, in_scene = {}, {}
    for model in models:
        v = load_verdicts(model_dir(root, model))
        in_scene[model] = {k: x.scene_windows >= SCENE_AT for k, x in v.items()}
        share = sum(in_scene[model].values()) / max(1, len(v))
        summary[model] = {"judged": len(v), "missing": len(expected - v.keys()), "scene_share": share}
        print(f"[{model}] {len(v)} of {len(expected)} clusters judged, {len(expected - v.keys())} missing; "
              f"{100 * share:.0f}% judged {emotion} scenes", flush=True)  # fmt: skip
    if len(models) == 2:
        a, b = (in_scene[m] for m in models)
        both = a.keys() & b.keys()
        summary["agreement"] = sum(a[k] == b[k] for k in both) / max(1, len(both))
        print(
            f"[agreement] the two models agree on {emotion} vs not for {100 * summary['agreement']:.0f}% of {len(both)} clusters"
        )
    return summary


def redo_incomplete(batches: list[Path], root: Path, models: list[str]) -> None:
    """Delete outputs with fewer verdicts than clusters, so the next pass asks for those files again."""
    for model in models:
        for batch in batches:
            out = model_dir(root, model) / batch.with_suffix(".jsonl").name
            if out.exists() and len(out.read_text().splitlines()) < len(cluster_keys(batch.read_text())):
                out.unlink()


def run_judge(cfg: ModesRunConfig) -> dict:
    """The judge stage: results/<run_id>/judgements/<model>/batch_XX.jsonl for every model in `judge.models`.

    Two passes: files with missing clusters are asked again once more. A judge that skips the same clusters
    every time leaves them missing; the report then counts only the judges that answered.
    """
    batches = sorted((cfg.result_dir / "modes").glob("batch_*.txt"))
    expected = {k for b in batches for k in cluster_keys(b.read_text())}
    root = cfg.result_dir / "judgements"
    key = read_key(cfg.judge.key_file)
    print(
        f"[judge] {len(batches)} batch files, {len(expected)} clusters, emotion {cfg.emotion}, models {cfg.judge.models}",
        flush=True,
    )
    for attempt in range(2):
        if attempt:
            redo_incomplete(batches, root, cfg.judge.models)
        for model in cfg.judge.models:
            judge_model(model, cfg.emotion, batches, root, key, cfg.judge.workers, cfg.judge.max_tokens)
    return judge_summary(cfg.judge.models, root, expected, cfg.emotion)
