"""A heatmap as template data: a grid of cells, each with its text, tooltip and colours for both modes.

- Sequential: the colour runs from the smallest value to the largest.
- Diverging: zero is gray, and the larger absolute value sets both ends, so equal sizes get equal colours.
- `marked` cells get an outline (e.g. the chosen candidate); `excluded` cells are hatched (e.g. filtered out).
- A missing value (None) is an empty cell.
- `log`: colours follow log10 of the value, for positive values that span orders of magnitude (e.g. KL). Text keeps the value.
"""

import math
from collections.abc import Callable, Sequence

from pydantic import BaseModel

from .colors import RAMPS, Hex, Scale, along, ink_for

Grid = Sequence[Sequence[float | None]]
Position = tuple[int, int]


class Cell(BaseModel):
    text: str
    tip: str
    fill_light: Hex | None
    fill_dark: Hex | None
    ink_light: Hex | None
    ink_dark: Hex | None
    marked: bool = False
    excluded: bool = False


class Heatmap(BaseModel):
    title: str
    caption: str
    row_title: str
    col_title: str
    row_labels: list[str]
    col_labels: list[str]
    rows: list[list[Cell]]
    legend_light: list[Hex]
    legend_dark: list[Hex]
    legend_low: str
    legend_high: str
    show_values: bool
    has_marked: bool
    has_excluded: bool


def present_values(grid: Grid) -> list[float]:
    present = []
    for row in grid:
        present += [v for v in row if v is not None]
    return present


def log10_row(row: Sequence[float | None]) -> list[float | None]:
    return [None if v is None else math.log10(v) for v in row]


def position_on_scale(scale: Scale, values: Grid) -> Callable[[float], float]:
    present = present_values(values)
    if scale == "diverging":
        span = max((abs(v) for v in present), default=1.0) or 1.0
        return lambda v: 0.5 + v / (2 * span)
    low, high = min(present, default=0.0), max(present, default=1.0)
    return lambda v: (v - low) / ((high - low) or 1.0)


def legend_ends(scale: Scale, values: Grid, fmt: str) -> tuple[str, str]:
    present = present_values(values)
    if scale == "diverging":
        span = max((abs(v) for v in present), default=1.0)
        return format(-span, fmt), format(span, fmt)
    return format(min(present), fmt), format(max(present), fmt)


def make_cell(value: float | None, t: Callable[[float], float], scale: Scale, fmt: str, tip: str) -> Cell:
    if value is None:
        return Cell(text="", tip=tip + ": none", fill_light=None, fill_dark=None, ink_light=None, ink_dark=None)
    light, dark = along(RAMPS[scale].light, t(value)), along(RAMPS[scale].dark, t(value))
    text = format(value, fmt)
    return Cell(
        text=text, tip=f"{tip}: {text}", fill_light=light, fill_dark=dark, ink_light=ink_for(light), ink_dark=ink_for(dark)
    )


def heatmap(
    title: str,
    values: Grid,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    *,
    scale: Scale,
    fmt: str = ".2f",
    caption: str = "",
    row_title: str = "",
    col_title: str = "",
    marked: set[Position] = frozenset(),
    excluded: set[Position] = frozenset(),
    show_values: bool = True,
    log: bool = False,
) -> Heatmap:
    assert len(values) == len(row_labels) and all(len(r) == len(col_labels) for r in values), "labels do not match the grid"
    t = position_on_scale(scale, values)
    if log:
        assert scale == "sequential", "a log scale is only for sequential values"
        logged = [log10_row(row) for row in values]
        t_log = position_on_scale(scale, logged)
        t = lambda v: t_log(math.log10(v))  # noqa: E731
    rows = []
    for i, row in enumerate(values):
        cells = []
        for j, v in enumerate(row):
            cell = make_cell(v, t, scale, fmt, f"{row_labels[i]} / {col_labels[j]}")
            cell.marked, cell.excluded = (i, j) in marked, (i, j) in excluded
            cells.append(cell)
        rows.append(cells)
    stops = [k / 8 for k in range(9)]
    low, high = legend_ends(scale, values, fmt)
    return Heatmap(
        title=title,
        caption=caption,
        row_title=row_title,
        col_title=col_title,
        row_labels=list(row_labels),
        col_labels=list(col_labels),
        rows=rows,
        legend_light=[along(RAMPS[scale].light, s) for s in stops],
        legend_dark=[along(RAMPS[scale].dark, s) for s in stops],
        legend_low=low,
        legend_high=high,
        show_values=show_values,
        has_marked=bool(marked),
        has_excluded=bool(excluded),
    )
