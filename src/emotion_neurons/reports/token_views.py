"""Token views: circuitsvis strips of per-token scores (`token_scores`), as HTML snippets for the templates.

- Emotion stories: `colored_tokens`, one vector's cosine, on a scale shared by that vector's views.
- Direction prompts: `colored_tokens`, the projection on one shared scale, prompt and reply as two strips.
- circuitsvis loads its renderer from unpkg, so a report needs a network connection to show these views.
"""

from circuitsvis.tokens import colored_tokens
from jaxtyping import Float
from pydantic import BaseModel
from torch import Tensor

from ..token_scores import ScoreIndex, Scores, TokenText
from .colors import RAMPS


class Strip(BaseModel):
    label: str
    html: str


class TokenView(BaseModel):
    title: str
    subtitle: str
    strips: list[Strip]


def closed(html: str) -> str:
    """circuitsvis writes `<div .../>`. HTML does not self-close a div, so the rest of the page would end up inside it."""
    head, sep, rest = html.partition("/>")
    assert sep and head.lstrip().startswith("<div"), "unexpected circuitsvis markup"
    return f"{head}></div>{rest}"


def scores_of(text: TokenText, scores: Scores) -> Scores:
    return scores[text.start : text.start + len(text.tokens)]


def first_per_label(texts: list[TokenText], labels: list[str], n: int) -> list[TokenText]:
    first = []
    for label in labels:
        with_label = [t for t in texts if t.label == label]
        first += with_label[:n]
    return first


def diverging_strip(label: str, tokens: list[str], v: Float[Tensor, " token"], span: float) -> Strip:
    low, high = RAMPS["diverging"].light.low, RAMPS["diverging"].light.high
    html = colored_tokens(tokens, v, min_value=-span, max_value=span, negative_color=low, positive_color=high)
    return Strip(label=label, html=closed(str(html)))


def vector_view(text: TokenText, v: Float[Tensor, " token"], mean: float, span: float) -> TokenView:
    return TokenView(
        title=f"{text.id} ({text.label})", subtitle=f"mean {mean:.3f}", strips=[diverging_strip("", text.tokens, v, span)]
    )


def prompt_views(index: ScoreIndex, scores: Scores, labels: list[str], per_label: int) -> list[TokenView]:
    span = float(scores.abs().max())

    views = []
    for text in first_per_label(index.texts, labels, per_label):
        v, k = scores_of(text, scores), text.marker
        assert k is not None, f"{text.id} has no reply marker"
        views.append(
            TokenView(
                title=f"{text.id} ({text.label})",
                subtitle=f"Projection at the last prompt token: {v[k - 1, 0]:.2f}. Reply mean: {v[k:, 0].mean():.2f}.",
                strips=[
                    diverging_strip("prompt", text.tokens[:k], v[:k, 0], span),
                    diverging_strip("reply", text.tokens[k:], v[k:, 0], span),
                ],
            )
        )
    return views
