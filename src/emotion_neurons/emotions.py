"""The 12 emotions, in the row order of every per-emotion array."""

from typing import Literal

EMOTIONS: tuple[str, ...] = (
    "happy", "inspired", "loving", "proud", "calm", "desperate",
    "angry", "guilty", "sad", "afraid", "nervous", "surprised",
)  # fmt: skip

LABELS: tuple[str, ...] = (*EMOTIONS, "neutral")
"""Story labels: the 12 emotions, then neutral."""

Site = Literal["resid_mid", "resid_post", "mlp_act", "mlp_out"]

HOOKED_SITES: tuple[Site, ...] = ("resid_mid", "resid_post", "mlp_act")
"""Recorded with hooks.

- resid_mid: the input to post_attention_layernorm, the stream after attention.
- resid_post: the layer's output.
- mlp_act: the input to down_proj, silu(gate) * up.
"""

SITES: tuple[Site, ...] = (*HOOKED_SITES, "mlp_out")
"""mlp_out = resid_post - resid_mid, derived. Exact for pooled sums too, since pooling is linear."""

RANGES = ("early", "late")
"""early: tokens [0, split_token). late: tokens [split_token, end)."""


def emotion_index(label: str) -> int:
    """-1 for neutral."""
    return EMOTIONS.index(label) if label in EMOTIONS else -1
