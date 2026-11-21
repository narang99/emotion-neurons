"""A page of per-token cosines with two vectors, side by side, drawn with circuitsvis token strips.

- Input: results/<run_id>/token_scores.json (scripts/token_scores.py) and removal.json (scripts/theme_removal.py).
- Shown: every held-out story of the emotion, and the other stories whose mean cosine changes most.
  - Per story: the original vector | the theme-removed vector, on one shared colour scale; below, the difference.
- Summary: mean token cosine per story label for both vectors, and the tokens that lose the most cosine.
"""

import html
import json
import re
from collections import defaultdict
from pathlib import Path

from circuitsvis.tokens import colored_tokens

STYLE = """
:root { --bg: #f7f7f5; --panel: #ffffff; --fg: #1f2421; --muted: #5f6b65; --line: #dfe3df; --accent: #8a3b12;
  --sans: "IBM Plex Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #131614; --panel: #1b201d; --fg: #e5e9e6;
  --muted: #9ba8a1; --line: #2d3531; --accent: #f0a070; color-scheme: dark } }
:root[data-theme="dark"] { --bg: #131614; --panel: #1b201d; --fg: #e5e9e6; --muted: #9ba8a1; --line: #2d3531;
  --accent: #f0a070; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--sans); font-size: 14px; line-height: 1.5; }
.wrap { max-width: 1200px; margin: 0 auto; padding-block: 24px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 18px; }
h1 { font-size: 22px; margin: 0; text-wrap: balance; } h2 { font-size: 17px; margin: 8px 0 0; }
p { margin: 0; max-width: 80ch; color: var(--muted); }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }
th, td { padding: 4px 12px 4px 0; border-bottom: 1px solid var(--line); text-align: right; } th:first-child, td:first-child { text-align: left; }
.table-wrap { overflow-x: auto; }
.story { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; display: grid; gap: 8px; min-width: 0; }
.story-head { display: flex; flex-wrap: wrap; gap: 4px 14px; align-items: baseline; }
.sid { font-family: var(--mono); font-weight: 500; } .meta { color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; min-width: 0; }
.col { min-width: 0; overflow-x: auto; } .label { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); font-weight: 600; }
.tok { font-family: var(--mono); font-size: 12px; padding: 0 5px; border: 1px solid var(--line); border-radius: 4px; white-space: pre; }
@media (max-width: 760px) { .pair { grid-template-columns: 1fr; } }
"""


def mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def strip(tokens: list[str], values: list[float], scale: float) -> str:
    """circuitsvis writes its container as `<div .../>`. HTML has no self-closing div, so each strip would stay open
    and every later story would nest inside it, narrower each time. Close it explicitly."""
    rendered = str(colored_tokens(tokens, values, min_value=-scale, max_value=scale))
    return re.sub(r'(<div id="circuits-vis-[^"]+"[^>]*?)\s*/>', r"\1></div>", rendered, count=1)


def story_block(s: dict, scale: float, diff_scale: float, split: str | None = None) -> str:
    diff = [b - a for a, b in zip(s["d"], s["d_minus"])]
    worst = sorted(zip(diff, s["tokens"]), key=lambda x: x[0])[:6]
    chips = "".join(f'<span class="tok">{html.escape(t.strip() or repr(t))} {v:+.3f}</span> ' for v, t in worst)
    return f"""<div class="story">
  <div class="story-head"><span class="sid">{html.escape(s["id"])}</span><span class="meta">label: {html.escape(s["label"])}{" · " + split if split else ""} ·
    mean cosine {mean(s["d"]):+.3f} → {mean(s["d_minus"]):+.3f} ({mean(diff):+.4f})</span></div>
  <div class="pair">
    <div class="col"><div class="label">original vector</div>{strip(s["tokens"], s["d"], scale)}</div>
    <div class="col"><div class="label">theme removed</div>{strip(s["tokens"], s["d_minus"], scale)}</div>
  </div>
  <div class="col"><div class="label">difference (removed − original); tokens that lose most: {chips}</div>{strip(s["tokens"], diff, diff_scale)}</div>
</div>"""


def summary_tables(stories: list[dict]) -> str:
    by_label = defaultdict(lambda: {"d": [], "d_minus": []})
    for s in stories:
        by_label[s["label"]]["d"] += s["d"]
        by_label[s["label"]]["d_minus"] += s["d_minus"]
    rows = sorted(by_label.items(), key=lambda kv: -mean(kv[1]["d"]))
    label_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{mean(v['d']):+.4f}</td><td>{mean(v['d_minus']):+.4f}</td><td>{mean(v['d_minus']) - mean(v['d']):+.4f}</td><td>{len(v['d'])}</td></tr>"
        for k, v in rows
    )
    drops = defaultdict(list)
    for s in stories:
        for t, a, b in zip(s["tokens"], s["d"], s["d_minus"]):
            drops[t.strip().lower()].append(b - a)
    common = [(t, mean(v), len(v)) for t, v in drops.items() if len(v) >= 10 and t]
    token_rows = "".join(
        f"<tr><td>{html.escape(t)}</td><td>{m:+.4f}</td><td>{n}</td></tr>" for t, m, n in sorted(common, key=lambda x: x[1])[:20]
    )
    return f"""<h2>Mean token cosine per story label</h2>
<div class="table-wrap"><table><tr><th>label</th><th>original</th><th>theme removed</th><th>change</th><th>tokens</th></tr>{label_rows}</table></div>
<h2>Tokens that lose the most cosine (seen 10+ times, all held-out stories)</h2>
<div class="table-wrap"><table><tr><th>token</th><th>mean change</th><th>count</th></tr>{token_rows}</table></div>"""


def build_page(result_dir: Path, others: int = 10) -> str:
    data = json.loads((result_dir / "token_scores.json").read_text())
    removal = json.loads((result_dir / "removal.json").read_text())
    emotion, stories = data["emotion"], data["stories"]
    own = [s for s in stories if s["label"] == emotion]
    rest = sorted(
        (s for s in stories if s["label"] != emotion), key=lambda s: mean([b - a for a, b in zip(s["d"], s["d_minus"])])
    )
    shown = own + rest[:others]
    scale = max(max(abs(x) for x in s[k]) for s in shown for k in ("d", "d_minus"))
    diff_scale = max(max(abs(b - a) for a, b in zip(s["d"], s["d_minus"])) for s in shown)
    sep = (removal["d"]["separation"], removal["d_minus"]["separation"])
    head = f"""<title>{emotion.capitalize()} Minus {html.escape(removal["theme"])}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{STYLE}</style>"""
    intro = f"""<h1>The {emotion} vector with and without its "{html.escape(removal["theme"])}" neurons</h1>
<p>Layer-25 MLP write of Qwen3-8B. The original vector is built from train stories; the second has {len(removal["removed"])} neurons
removed ({removal["removed_push_pct"]:.2f}% of the push): those whose only {emotion} theme is "{html.escape(removal["theme"])}".
The two vectors point the same way at cosine {removal["cos_d_d_minus"]:.3f}. Held-out separation (an {emotion} story scores above a
non-{emotion} one): {100 * sep[0]:.2f}% → {100 * sep[1]:.2f}%.</p>
<p>Each token is coloured by the cosine of its MLP output (minus the mean train output) with the vector: same scale for both
columns (±{scale:.2f}). The difference strip has its own scale (±{diff_scale:.3f}). Shown: the {len(own)} held-out {emotion} stories,
then the {others} other held-out stories whose mean cosine drops most.</p>"""
    blocks = "\n".join(story_block(s, scale, diff_scale) for s in shown)
    return f'{head}\n<div class="wrap">{intro}{summary_tables(stories)}<h2>Stories</h2>{blocks}</div>'


def story_changes(s: dict) -> dict:
    """Per story: the biggest single-token drop (and its token) and the total change, removed minus original."""
    diff = [b - a for a, b in zip(s["d"], s["d_minus"])]
    i = min(range(len(diff)), key=diff.__getitem__)
    return {"max_drop": diff[i], "max_drop_token": s["tokens"][i], "total": sum(diff)}


def ranking_section(title: str, key: str, stories: list[dict], changes: dict, top_table: int, top_shown: int) -> str:
    ranked = sorted(stories, key=lambda s: changes[s["id"]][key])
    labels = defaultdict(int)
    for s in ranked[:top_table]:
        labels[s["label"]] += 1
    spread = ", ".join(f"{k} {v}" for k, v in sorted(labels.items(), key=lambda kv: -kv[1]))
    rows = "".join(
        f"<tr><td>{r}</td><td>{html.escape(s['id'])}</td><td>{html.escape(s['label'])}</td><td>{s.get('split', 'heldout')}</td>"
        f"<td>{changes[s['id']]['max_drop']:+.3f}</td><td>{html.escape(changes[s['id']]['max_drop_token'].strip())}</td>"
        f"<td>{changes[s['id']]['total']:+.2f}</td></tr>"
        for r, s in enumerate(ranked[:top_table], 1)
    )
    shown = ranked[:top_shown]
    scale = max(max(abs(x) for x in s[k]) for s in shown for k in ("d", "d_minus"))
    diff_scale = max(max(abs(b - a) for a, b in zip(s["d"], s["d_minus"])) for s in shown)
    blocks = "\n".join(story_block(s, scale, diff_scale, split=s.get("split", "heldout")) for s in shown)
    return f"""<h2>{title}</h2>
<p>Top {top_table} by emotion: {spread}.</p>
<div class="table-wrap"><table><tr><th>rank</th><th>story</th><th>label</th><th>split</th><th>biggest token drop</th><th>token</th><th>total change</th></tr>{rows}</table></div>
<p>The top {top_shown}, drawn side by side (colour scale ±{scale:.2f}; difference ±{diff_scale:.3f}).</p>
{blocks}"""


def build_ranked_page(
    result_dir: Path, scores_file: str = "token_scores_all.json", top_table: int = 50, top_shown: int = 25
) -> str:
    data = json.loads((result_dir / scores_file).read_text())
    removal = json.loads((result_dir / "removal.json").read_text())
    emotion, stories = data["emotion"], data["stories"]
    changes = {s["id"]: story_changes(s) for s in stories}
    head = f"""<title>{emotion.capitalize()} Minus {html.escape(removal["theme"])}: Rankings</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{STYLE}</style>"""
    intro = f"""<h1>Where removing the "{html.escape(removal["theme"])}" neurons changes the {emotion} vector most</h1>
<p>All {len(stories)} stories (train and held-out; the train stories built the vectors, so they show which tokens move,
not how well the vector generalises). Change = cosine with the theme-removed vector minus cosine with the original, per
token ({len(removal["removed"])} neurons removed, {removal["removed_push_pct"]:.2f}% of the push). Two rankings, biggest drop first.</p>"""
    sections = [
        ranking_section("Ranked by the biggest single-token drop", "max_drop", stories, changes, top_table, top_shown),
        ranking_section("Ranked by the total change over the story", "total", stories, changes, top_table, top_shown),
    ]
    return f'{head}\n<div class="wrap">{intro}{"".join(sections)}</div>'
