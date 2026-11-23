"""Heatmap cell colours, one pair (light, dark) per cell.

- Sequential: one blue hue, near the surface at the low end, dark (light mode) or bright (dark mode) at the high end.
- Diverging: blue for negative, red for positive, a neutral gray at zero. Equal steps per arm.
- Colours mix in OKLab, so lightness changes evenly along a ramp. End points are the dataviz reference palette's.
"""

from typing import Literal, NamedTuple

Hex = str
Lab = tuple[float, float, float]
Scale = Literal["sequential", "diverging"]


class Ramp(NamedTuple):
    low: Hex
    mid: Hex
    high: Hex


class Pair(NamedTuple):
    light: Hex
    dark: Hex


RAMPS: dict[Scale, Pair] = {
    "sequential": Pair(Ramp("#eef5fd", "#6da7ec", "#0d366b"), Ramp("#1f2a38", "#3987e5", "#cde2fb")),
    "diverging": Pair(Ramp("#184f95", "#f0efec", "#a8302f"), Ramp("#6da7ec", "#383835", "#e66767")),
}
DARK_INK, LIGHT_INK = "#0b0b0b", "#f7f7f5"


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> float:
    c = min(max(c, 0.0), 1.0)
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def hex_to_oklab(h: Hex) -> Lab:
    r, g, b = (srgb_to_linear(int(h[i : i + 2], 16) / 255) for i in (1, 3, 5))
    l_, m_, s_ = (
        (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3),
        (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3),
        (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3),
    )
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def oklab_to_hex(lab: Lab) -> Hex:
    L, a, b = lab
    l_, m_, s_ = (
        (L + 0.3963377774 * a + 0.2158037573 * b) ** 3,
        (L - 0.1055613458 * a - 0.0638541728 * b) ** 3,
        (L - 0.0894841775 * a - 1.2914855480 * b) ** 3,
    )
    rgb = (
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    )
    return "#" + "".join(f"{round(linear_to_srgb(c) * 255):02x}" for c in rgb)


def mix(a: Hex, b: Hex, t: float) -> Hex:
    la, lb = hex_to_oklab(a), hex_to_oklab(b)
    return oklab_to_hex(tuple(x + (y - x) * t for x, y in zip(la, lb)))


def along(ramp: Ramp, t: float) -> Hex:
    """t in [0, 1]: low -> mid over the first half, mid -> high over the second."""
    t = min(max(t, 0.0), 1.0)
    return mix(ramp.low, ramp.mid, 2 * t) if t < 0.5 else mix(ramp.mid, ramp.high, 2 * t - 1)


def lightness(h: Hex) -> float:
    return hex_to_oklab(h)[0]


def ink_for(fill: Hex) -> Hex:
    """Dark text on light fills, light text on dark fills, in either mode."""
    return DARK_INK if lightness(fill) >= 0.62 else LIGHT_INK
