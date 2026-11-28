"""Line charts of the refusal direction under ablation, against the first ablated layer, as one HTML page. CPU, seconds.

    python experiments/refusal-ablation/plot.py

- Reads refusal_ablation/results.json next to this script; writes report.html next to this script.
- Charts: cosine with the intact direction, and the ablated model's refusal rate on harmful and on harmless prompts.
- Colour is the band (p3/p97, p5/p95); a solid line is the band ablation, a dashed line the random control.
- Reference lines: the split-half floor (mean and min) on the cosine chart, the intact model's rate on the refusal charts.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "refusal_ablation" / "results.json"
WIDTH, HEIGHT = 720, 280
LEFT, RIGHT, TOP, BOTTOM = 56, 150, 16, 40
TICK_STEPS = (0.05, 0.1, 0.125, 0.25)


def series_of(conditions: list[dict], metric) -> dict[tuple[str, str], list[tuple[int, float]]]:
    """(band, kind) -> [(start, value)], sorted by start."""
    out: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for c in conditions:
        cond = c["condition"]
        if cond["band"] is not None:
            out.setdefault((cond["band"], cond["kind"]), []).append((cond["start"], metric(c)))
    return {k: sorted(v) for k, v in out.items()}


def all_points(series: dict[tuple[str, str], list[tuple[int, float]]]) -> list[tuple[int, float]]:
    """Every (start, value) point of every series."""
    points = []
    for series_points in series.values():
        points.extend(series_points)
    return points


def axis_range(values: list[float]) -> tuple[float, float, float]:
    """(bottom, top, step): four or fewer gridline steps that cover the values, from a step multiple below them."""
    low, high = min(values), max(values)
    for step in TICK_STEPS:
        bottom = step * int(low / step - 1e-9) if low > 0 else 0.0
        top = bottom + 4 * step
        if top >= high:
            return bottom, top, step
    return 0.0, 1.0, 0.25


def dodged(ys: list[float], gap: float = 12) -> list[float]:
    """Label heights pushed apart to at least `gap`, so labels of close reference lines do not overlap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for before, after in zip(order, order[1:]):
        out[after] = max(out[after], out[before] + gap)
    return out


def chart(
    title: str,
    series: dict[tuple[str, str], list[tuple[int, float]]],
    references: dict[str, float],
    last_layer: int,
    fixed: tuple[float, float, float] | None = None,
) -> str:
    bands = sorted({band for band, _ in series})
    values = [v for _, v in all_points(series)] + list(references.values())
    bottom, top, step = fixed or axis_range(values)
    x = lambda k: LEFT + k * (WIDTH - LEFT - RIGHT) / last_layer  # noqa: E731
    y = lambda v: TOP + (1 - (v - bottom) / (top - bottom)) * (HEIGHT - TOP - BOTTOM)  # noqa: E731
    ticks = [bottom + i * step for i in range(int(round((top - bottom) / step)) + 1)]
    grid = "".join(
        f'<line class="grid" x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
        f'<text class="tick" x="{LEFT - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:g}</text>'
        for v in ticks
    )
    starts = sorted({k for k, _ in all_points(series)})
    xticks = "".join(
        f'<text class="tick" x="{x(k):.1f}" y="{HEIGHT - BOTTOM + 18}" text-anchor="middle">{k}</text>' for k in starts
    )
    label_ys = dodged([y(v) + 4 for v in references.values()])
    refs = "".join(
        f'<line class="ref" x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
        f'<text class="direct" x="{WIDTH - RIGHT + 8}" y="{ly:.1f}">{name} {v:.3f}</text>'
        for (name, v), ly in zip(references.items(), label_ys)
    )
    lines = ""
    for (band, kind), points in series.items():
        colour = bands.index(band) + 1
        coords = " ".join(f"{x(k):.1f},{y(v):.1f}" for k, v in points)
        lines += f'<polyline class="line s{colour} {kind}" points="{coords}"/>'
        lines += "".join(f'<circle class="dot s{colour}" cx="{x(k):.1f}" cy="{y(v):.1f}" r="3"/>' for k, v in points)
    data = json.dumps({f"{band} {kind}": dict(points) for (band, kind), points in series.items()})
    return f"""<figure class="chart" data-chart='{data}'>
  <figcaption>{title}</figcaption>
  <svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{title} against the first ablated layer">
    {grid}{xticks}{refs}
    <text class="axis" x="{(LEFT + WIDTH - RIGHT) / 2}" y="{HEIGHT - 4}" text-anchor="middle">first ablated layer k (layers k to {last_layer} ablated)</text>
    {lines}
    <line class="cross" y1="{TOP}" y2="{HEIGHT - BOTTOM}" visibility="hidden"/>
    <rect class="hit" x="{LEFT - 10}" y="{TOP}" width="{WIDTH - LEFT - RIGHT + 20}" height="{HEIGHT - TOP - BOTTOM}" data-left="{LEFT}" data-span="{WIDTH - LEFT - RIGHT}" data-last="{last_layer}" data-starts='{json.dumps(starts)}'/>
  </svg>
  <div class="tip" hidden></div>
</figure>"""


def table(conditions: list[dict]) -> str:
    head = "<tr><th>condition</th><th>cells ablated</th><th>cosine</th><th>norm</th><th>refusal, harmful</th><th>refusal, harmless</th></tr>"
    rows = ""
    for c in conditions:
        frac = "" if c["ablated_fraction"] is None else f"{c['ablated_fraction']:.1%}"
        rates = c["refusal_rates"]
        rows += f"<tr><td>{c['condition']['name']}</td><td>{frac}</td><td>{c['cosine']:.4f}</td><td>{c['norm']:.2f}</td><td>{rates['harmful']:.3f}</td><td>{rates['harmless']:.3f}</td></tr>"
    return f'<details><summary>Values as a table</summary><div class="scroll"><table>{head}{rows}</table></div></details>'


STYLE = """
.viz-root { color-scheme: light; --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
  --muted: #8a8984; --rule: #e3e2de; --series-1: #2a78d6; --series-2: #eb6834; }
@media (prefers-color-scheme: dark) { .viz-root { color-scheme: dark; --surface-1: #1a1a19; --text-primary: #f0efec;
  --text-secondary: #c3c2b7; --muted: #8a8984; --rule: #34332f; --series-1: #3987e5; --series-2: #d95926; } }
body { margin: 0; }
.viz-root { background: var(--surface-1); color: var(--text-primary); font: 14px/1.4 system-ui, sans-serif; padding: 24px 32px; min-height: 100vh; }
h1 { font-size: 18px; margin: 0 0 12px; }
.legend { display: flex; gap: 16px; color: var(--text-secondary); font-size: 13px; margin-bottom: 8px; align-items: center; }
.legend svg { width: 24px; height: 8px; display: inline-block; vertical-align: middle; margin-right: 6px; }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 24px; }
.chart { position: relative; margin: 0; }
.chart figcaption { font-weight: 600; margin-bottom: 4px; }
.chart svg { width: 100%; height: auto; display: block; }
.grid { stroke: var(--rule); stroke-width: 1; }
.tick, .axis { fill: var(--muted); font-size: 11px; }
.ref { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 2 3; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; }
.line.random { stroke-dasharray: 6 4; }
.s1 { stroke: var(--series-1); } .s2 { stroke: var(--series-2); }
.dot { fill: var(--surface-1); stroke-width: 2; }
.direct { font-size: 11px; fill: var(--text-secondary); }
.cross { stroke: var(--muted); stroke-width: 1; }
.hit { fill: transparent; }
.tip { position: absolute; pointer-events: none; background: var(--surface-1); border: 1px solid var(--rule); border-radius: 6px;
  padding: 6px 8px; font-size: 12px; color: var(--text-primary); box-shadow: 0 2px 6px rgba(0,0,0,.12); white-space: nowrap; }
details { margin-top: 24px; color: var(--text-secondary); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 12px; color: var(--text-primary); }
th, td { padding: 3px 8px; border-bottom: 1px solid var(--rule); text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }
"""

HOVER = """
for (const fig of document.querySelectorAll(".chart")) {
  const d = JSON.parse(fig.dataset.chart), svg = fig.querySelector("svg"), hit = fig.querySelector(".hit");
  const cross = fig.querySelector(".cross"), tip = fig.querySelector(".tip");
  const left = +hit.dataset.left, span = +hit.dataset.span, last = +hit.dataset.last, starts = JSON.parse(hit.dataset.starts);
  hit.addEventListener("mousemove", e => {
    const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY;
    const local = p.matrixTransform(svg.getScreenCTM().inverse());
    const at = (local.x - left) / span * last;
    const k = starts.reduce((a, b) => Math.abs(b - at) < Math.abs(a - at) ? b : a);
    const cx = left + k * span / last;
    cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("visibility", "visible");
    tip.innerHTML = `layers ${k} to ${last}<br>` + Object.entries(d).map(([n, v]) => `${n}: ${(+v[k]).toFixed(3)}`).join("<br>");
    const box = fig.getBoundingClientRect();
    tip.hidden = false;
    tip.style.left = Math.min(e.clientX - box.left + 12, box.width - tip.offsetWidth) + "px";
    tip.style.top = (e.clientY - box.top + 12) + "px";
  });
  hit.addEventListener("mouseleave", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
}
"""


def legend(bands: list[str]) -> str:
    def swatch(colour: int, dashed: bool) -> str:
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        return f'<svg viewBox="0 0 24 8"><line x1="0" x2="24" y1="4" y2="4" stroke="var(--series-{colour})" stroke-width="2"{dash}/></svg>'

    items = [
        f"<span>{swatch(i, False)}{band} band</span><span>{swatch(i, True)}{band} random</span>"
        for i, band in enumerate(bands, start=1)
    ]
    return f'<div class="legend">{"".join(items)}</div>'


def main():
    r = json.loads(RESULTS.read_text())
    conditions = r["conditions"]
    intact = next(c for c in conditions if c["condition"]["name"] == "intact")
    last_layer = r["site"]["layer"] - 1
    bands = sorted({c["condition"]["band"] for c in conditions if c["condition"]["band"]})
    floor = r["split_half_floor"]
    rate = (0.0, 1.0, 0.25)
    charts = (
        chart(
            "Cosine with the intact direction",
            series_of(conditions, lambda c: c["cosine"]),
            {"split-half mean": floor["mean"], "split-half min": floor["min"]},
            last_layer,
        )
        + chart(
            "Refusal rate, harmful prompts",
            series_of(conditions, lambda c: c["refusal_rates"]["harmful"]),
            {"intact": intact["refusal_rates"]["harmful"]},
            last_layer,
            rate,
        )
        + chart(
            "Refusal rate, harmless prompts",
            series_of(conditions, lambda c: c["refusal_rates"]["harmless"]),
            {"intact": intact["refusal_rates"]["harmless"]},
            last_layer,
            rate,
        )
    )
    title = f"Refusal direction of Qwen-1 7B Chat (layer {r['site']['layer']} input, position {r['site']['position']}) with layers k to {last_layer} mean-ablated"
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{STYLE}</style></head>
<body><main class="viz-root">
<h1>{title}</h1>
{legend(bands)}
<div class="grid-2">{charts}</div>
{table(conditions)}
</main><script>{HOVER}</script></body></html>"""
    (HERE / "report.html").write_text(page)
    print(f"[saved] {HERE / 'report.html'}")


if __name__ == "__main__":
    main()
