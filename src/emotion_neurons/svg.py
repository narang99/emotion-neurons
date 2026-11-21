"""Small line charts as SVG text, for reports written in batch (no matplotlib)."""

from collections.abc import Sequence

PALETTE = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4b5563")
WIDTH, HEIGHT, PAD = 640, 300, 44


def scale(value: float, low: float, high: float, out_low: float, out_high: float) -> float:
    return out_low + (value - low) / (high - low or 1.0) * (out_high - out_low)


def polyline(values: Sequence[float], y_low: float, y_high: float, color: str) -> str:
    n = len(values)
    points = " ".join(
        f"{scale(i, 0, n - 1, PAD, WIDTH - PAD):.1f},{scale(v, y_low, y_high, HEIGHT - PAD, PAD):.1f}"
        for i, v in enumerate(values)
    )
    return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>'


def axes(n: int, y_low: float, y_high: float, x_label: str) -> list[str]:
    parts = [f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>']
    for value in (y_low, (y_low + y_high) / 2, y_high, 0.0):
        if y_low <= value <= y_high:
            y = scale(value, y_low, y_high, HEIGHT - PAD, PAD)
            parts.append(f'<line x1="{PAD}" x2="{WIDTH - PAD}" y1="{y:.1f}" y2="{y:.1f}" stroke="#ddd"/>')
            parts.append(f'<text x="{PAD - 6}" y="{y + 4:.1f}" font-size="11" text-anchor="end">{value:.2f}</text>')
    for i in range(0, n, max(1, n // 9)):
        x = scale(i, 0, n - 1, PAD, WIDTH - PAD)
        parts.append(f'<text x="{x:.1f}" y="{HEIGHT - PAD + 16}" font-size="11" text-anchor="middle">{i}</text>')
    parts.append(f'<text x="{WIDTH / 2}" y="{HEIGHT - 8}" font-size="12" text-anchor="middle">{x_label}</text>')
    return parts


def legend(names: Sequence[str]) -> list[str]:
    return [
        f'<text x="{PAD + 8 + 110 * i}" y="{PAD - 12}" font-size="12" fill="{PALETTE[i % len(PALETTE)]}">{name}</text>'
        for i, name in enumerate(names)
    ]


def line_chart(title: str, series: dict[str, Sequence[float]], x_label: str = "layer") -> str:
    values = [v for line in series.values() for v in line]
    y_low, y_high = min(values + [0.0]), max(values + [0.0])
    n = max(len(line) for line in series.values())
    parts = axes(n, y_low, y_high, x_label)
    parts += [polyline(line, y_low, y_high, PALETTE[i % len(PALETTE)]) for i, line in enumerate(series.values())]
    parts += legend(list(series))
    parts.append(f'<text x="{PAD}" y="18" font-size="14" font-weight="bold">{title}</text>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}">{"".join(parts)}</svg>'
