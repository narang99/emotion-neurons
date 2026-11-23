"""Line charts of the sweep metrics against the first ablated layer, as one self-contained HTML page. CPU, seconds.

    python experiments/goodfire-sparsity/plot.py

- Reads the ablation.json files of the full and sweep runs of both bands, next to this script.
- Writes report.html next to this script: loss change, text KL, harmful-prompt KL and harmless-prompt KL.
- Hover a chart for the values at a layer; "Values as a table" lists every number.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
BANDS = {"p3 / p97": ("ablation_full", "ablation_sweep"), "p5 / p95": ("ablation_full_p5", "ablation_sweep_p5")}
METRICS = {
    "Loss change (nats)": lambda a: a["loss_change"],
    "KL per position, held-out text": lambda a: a["kl_per_position"],
    "KL per position, harmful prompts": lambda a: a["prompts"]["harmful"]["kl_per_position"],
    "KL per position, harmless prompts": lambda a: a["prompts"]["harmless"]["kl_per_position"],
}
WIDTH, HEIGHT = 720, 280
LEFT, RIGHT, TOP, BOTTOM = 56, 72, 16, 40


def band_rows(full: str, sweep: str) -> list[dict]:
    """The band ablation of all layers, then layers k to 31 for k = 1 to 31."""
    rows = [a for a in json.loads((HERE / full / "ablation.json").read_text())["ablations"] if a["name"].startswith("band")]
    rows += json.loads((HERE / sweep / "ablation.json").read_text())["ablations"]
    assert [r["layers"][0] for r in rows] == list(range(len(rows))), "sweep rows are not k = 0, 1, 2, ..."
    return rows


TICK_STEPS = (0.025, 0.05, 0.1, 0.125, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3)


def axis_top(value: float) -> float:
    """Four gridline steps that cover `value`, with the smallest step that does."""
    return 4 * next(s for s in TICK_STEPS if 4 * s >= value)


def dodged(ys: list[float], gap: float = 12) -> list[float]:
    """Label heights pushed apart to at least `gap`, so end labels of close lines do not overlap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for before, after in zip(order, order[1:]):
        out[after] = max(out[after], out[before] + gap)
    return out


def chart(title: str, series: dict[str, list[float]]) -> str:
    layers = len(next(iter(series.values())))
    top = axis_top(max(max(v) for v in series.values()))
    x = lambda k: LEFT + k * (WIDTH - LEFT - RIGHT) / (layers - 1)  # noqa: E731
    y = lambda v: TOP + (1 - v / top) * (HEIGHT - TOP - BOTTOM)  # noqa: E731
    grid = "".join(
        f'<line class="grid" x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y(top * i / 4):.1f}" y2="{y(top * i / 4):.1f}"/>'
        f'<text class="tick" x="{LEFT - 8}" y="{y(top * i / 4) + 4:.1f}" text-anchor="end">{top * i / 4:g}</text>'
        for i in range(5)
    )
    xticks = "".join(
        f'<text class="tick" x="{x(k):.1f}" y="{HEIGHT - BOTTOM + 18}" text-anchor="middle">{k}</text>'
        for k in range(0, layers, 4)
    )
    lines = "".join(
        f'<polyline class="line s{i}" points="{" ".join(f"{x(k):.1f},{y(v):.1f}" for k, v in enumerate(values))}"/>'
        for i, values in enumerate(series.values(), start=1)
    )
    label_ys = dodged([y(values[-1]) + 4 for values in series.values()])
    labels = "".join(
        f'<text class="direct" x="{x(layers - 1) + 8:.1f}" y="{ly:.1f}">{name}</text>' for name, ly in zip(series, label_ys)
    )
    data = json.dumps({"names": list(series), "values": list(series.values()), "top": top, "layers": layers})
    return f"""<figure class="chart" data-chart='{data}'>
  <figcaption>{title}</figcaption>
  <svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{title} against the first ablated layer">
    {grid}{xticks}
    <text class="axis" x="{(LEFT + WIDTH - RIGHT) / 2}" y="{HEIGHT - 4}" text-anchor="middle">first ablated layer (layers k to 31 ablated)</text>
    {lines}{labels}
    <line class="cross" y1="{TOP}" y2="{HEIGHT - BOTTOM}" visibility="hidden"/>
    <rect class="hit" x="{LEFT}" y="{TOP}" width="{WIDTH - LEFT - RIGHT}" height="{HEIGHT - TOP - BOTTOM}"/>
  </svg>
  <div class="tip" hidden></div>
</figure>"""


def table(rows: dict[str, list[dict]]) -> str:
    head = "".join(f"<th>{band}: {metric}</th>" for metric in METRICS for band in rows)
    body = ""
    for k in range(len(next(iter(rows.values())))):
        cells = "".join(f"<td>{fn(rows[band][k]):.3f}</td>" for fn in METRICS.values() for band in rows)
        body += f"<tr><td>{k}</td>{cells}</tr>"
    return f'<details><summary>Values as a table</summary><div class="scroll"><table><tr><th>first ablated layer</th>{head}</tr>{body}</table></div></details>'


STYLE = """
.viz-root { color-scheme: light; --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
  --muted: #8a8984; --rule: #e3e2de; --series-1: #2a78d6; --series-2: #eb6834; }
@media (prefers-color-scheme: dark) { .viz-root { color-scheme: dark; --surface-1: #1a1a19; --text-primary: #f0efec;
  --text-secondary: #c3c2b7; --muted: #8a8984; --rule: #34332f; --series-1: #3987e5; --series-2: #d95926; } }
body { margin: 0; }
.viz-root { background: var(--surface-1); color: var(--text-primary); font: 14px/1.4 system-ui, sans-serif; padding: 24px 32px; min-height: 100vh; }
h1 { font-size: 18px; margin: 0 0 12px; }
.legend { display: flex; gap: 16px; color: var(--text-secondary); font-size: 13px; margin-bottom: 8px; }
.legend i { display: inline-block; width: 16px; height: 2px; vertical-align: middle; margin-right: 6px; }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 24px; }
.chart { position: relative; margin: 0; }
.chart figcaption { font-weight: 600; margin-bottom: 4px; }
svg { width: 100%; height: auto; display: block; }
.grid { stroke: var(--rule); stroke-width: 1; }
.tick, .axis { fill: var(--muted); font-size: 11px; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; }
.s1 { stroke: var(--series-1); } .s2 { stroke: var(--series-2); }
.direct { font-size: 11px; fill: var(--text-secondary); }
.cross { stroke: var(--muted); stroke-width: 1; }
.hit { fill: transparent; }
.tip { position: absolute; pointer-events: none; background: var(--surface-1); border: 1px solid var(--rule); border-radius: 6px;
  padding: 6px 8px; font-size: 12px; color: var(--text-primary); box-shadow: 0 2px 6px rgba(0,0,0,.12); white-space: nowrap; }
.tip i { display: inline-block; width: 10px; height: 2px; vertical-align: middle; margin-right: 6px; }
details { margin-top: 24px; color: var(--text-secondary); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 12px; color: var(--text-primary); }
th, td { padding: 3px 8px; border-bottom: 1px solid var(--rule); text-align: right; }
th { color: var(--text-secondary); font-weight: 600; }
"""

HOVER = """
for (const fig of document.querySelectorAll(".chart")) {
  const d = JSON.parse(fig.dataset.chart), svg = fig.querySelector("svg"), hit = fig.querySelector(".hit");
  const cross = fig.querySelector(".cross"), tip = fig.querySelector(".tip");
  const x0 = +hit.getAttribute("x"), w = +hit.getAttribute("width");
  hit.addEventListener("mousemove", e => {
    const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY;
    const local = p.matrixTransform(svg.getScreenCTM().inverse());
    const k = Math.max(0, Math.min(d.layers - 1, Math.round((local.x - x0) / w * (d.layers - 1))));
    const cx = x0 + k * w / (d.layers - 1);
    cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("visibility", "visible");
    tip.innerHTML = `layers ${k} to ${d.layers - 1}<br>` + d.names.map((n, i) =>
      `<i style="background:var(--series-${i + 1})"></i>${n}: ${d.values[i][k].toFixed(3)}`).join("<br>");
    const box = fig.getBoundingClientRect();
    tip.hidden = false;
    tip.style.left = Math.min(e.clientX - box.left + 12, box.width - tip.offsetWidth) + "px";
    tip.style.top = (e.clientY - box.top + 12) + "px";
  });
  hit.addEventListener("mouseleave", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
}
"""


def main():
    rows = {band: band_rows(*runs) for band, runs in BANDS.items()}
    charts = "".join(chart(title, {band: [fn(r) for r in rows[band]] for band in rows}) for title, fn in METRICS.items())
    legend = "".join(f'<span><i style="background:var(--series-{i})"></i>{band}</span>' for i, band in enumerate(rows, start=1))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Goodfire sparsity on Qwen-1 7B Chat: mean ablation of layers k to 31</title><style>{STYLE}</style></head>
<body><main class="viz-root">
<h1>Goodfire sparsity on Qwen-1 7B Chat: mean ablation of layers k to 31</h1>
<div class="legend">{legend}</div>
<div class="grid-2">{charts}</div>
{table(rows)}
</main><script>{HOVER}</script></body></html>"""
    (HERE / "report.html").write_text(page)
    print(f"[saved] {HERE / 'report.html'}")


if __name__ == "__main__":
    main()
