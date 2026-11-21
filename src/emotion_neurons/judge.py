"""Blind judgements of mode clusters by a cheap model on OpenRouter.

- Input: a modes batch file (`scripts/modes.py`): per neuron and pole, clusters of exemplar windows with the
  firing token marked [[like this]]. The judge sees no labels, no emotion and no reason the neuron was chosen.
- Per cluster the judge returns: a short mode name, how many windows sit in a happy or joyful scene, and the scenes.
- Scene, not word: a marked "and" inside a joyful scene counts as happy; the word "happy" in a sad scene does not.
- The API key is read from a file at run time and never printed.
"""

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from pydantic import BaseModel

API = "https://openrouter.ai/api/v1/chat/completions"

INSTRUCTIONS = """You read clusters of text windows. Each window has one token marked [[like this]]; a number in brackets before it is a neuron's activation there, which you can ignore.
The file is organised as: "##### neuron N", then "=== high pole" or "=== low pole", then "--- cluster K" followed by its windows. Ignore "--- unclustered (sample)" groups.

For every cluster, return:
- "mode": what the marked tokens and their contexts have in common, in at most 15 words;
- "happy_windows": how many of its windows sit in a happy or joyful SCENE. Judge the situation and mood of the surrounding text (laughing, celebrating, having fun, delight, warm happy moments). Do not judge by the marked word alone: a marked "and", "so", "." or "was" inside a joyful scene counts as happy; the word "happy" inside a sad or neutral scene does not. Recipes, technical text, news and neutral narration are not happy;
- "windows": how many windows the cluster has;
- "scene": what the scenes are, in 3 to 8 words.

Answer with one JSON object per line and nothing else, in file order, one line per cluster:
{"neuron": N, "pole": "high" or "low", "cluster": K, "mode": "...", "happy_windows": 0, "windows": 5, "scene": "..."}"""


class Verdict(BaseModel):
    neuron: int
    pole: str
    cluster: int
    mode: str
    happy_windows: int
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
            verdicts.append(Verdict(**json.loads(match.group(0))))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return verdicts


def ask(model: str, user: str, key: str, max_tokens: int, attempts: int = 4) -> str:
    """One chat completion; retries with backoff on network errors, rate limits and server errors."""
    body = json.dumps({
        "model": model, "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": user}],
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


def judge_batch(model: str, text: str, key: str, max_tokens: int) -> tuple[list[Verdict], list[tuple[int, str, int]]]:
    """Verdicts for every cluster of one batch file. Clusters missing from the first reply are asked again once."""
    expected = cluster_keys(text)
    found = {(v.neuron, v.pole, v.cluster): v for v in parse_verdicts(ask(model, text, key, max_tokens))}
    missing = [k for k in expected if k not in found]
    if missing:
        wanted = "\n".join(f"neuron {n}, {p} pole, cluster {c}" for n, p, c in missing)
        retry = f"{text}\n\nReturn lines only for these clusters, which were missing:\n{wanted}"
        for v in parse_verdicts(ask(model, retry, key, max_tokens)):
            found.setdefault((v.neuron, v.pole, v.cluster), v)
    verdicts = [found[k] for k in expected if k in found]
    return verdicts, [k for k in expected if k not in found]


def read_key(path: Path) -> str:
    key = path.expanduser().read_text().strip()
    assert key.startswith("sk-or-"), f"{path} does not hold an OpenRouter key"
    return key
