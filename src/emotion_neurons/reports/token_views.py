"""Token views: circuitsvis strips of per-token scores (`token_scores`), as HTML snippets for the templates.

- Emotion stories: `colored_tokens_multi`, one score per emotion; the reader picks the emotion that colours the text.
  - The story's own emotion comes first, so the view opens on it.
- Direction prompts: `colored_tokens`, the projection on one shared scale, prompt and reply as two strips.
- circuitsvis loads its renderer from unpkg, so a report needs a network connection to show these views.
"""

from circuitsvis.tokens import colored_tokens, colored_tokens_multi
from pydantic import BaseModel

from ..token_scores import Scores, TokenScores, TokenText
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


def scores_of(text: TokenText, values: Scores) -> Scores:
    return values[text.start : text.start + len(text.tokens)]


def first_per_label(texts: list[TokenText], labels: list[str], n: int) -> list[TokenText]:
    return [t for label in labels for t in [t for t in texts if t.label == label][:n]]


def own_first(series: list[str], label: str) -> list[int]:
    """Column order with the story's own emotion first: circuitsvis colours by the first label until the reader picks one."""
    own = [series.index(label)] if label in series else []
    return own + [i for i in range(len(series)) if i not in own]


def story_views(scores: TokenScores, values: Scores, labels: list[str], per_label: int, split_token: int) -> list[TokenView]:
    views = []
    for text in first_per_label(scores.texts, labels, per_label):
        v = scores_of(text, values)
        late = v[split_token:].mean(0)
        nearest = scores.series[int(late.argmax())]
        order = own_first(scores.series, text.label)
        views.append(TokenView(
            title=f"{text.id} ({text.label})",
            subtitle=f"Mean over tokens {split_token}+: nearest vector {nearest} ({late.max():.3f}); {text.label} {late[scores.series.index(text.label)]:.3f}."
            if text.label in scores.series else f"Mean over tokens {split_token}+: nearest vector {nearest} ({late.max():.3f}).",
            strips=[Strip(label="", html=closed(str(colored_tokens_multi(text.tokens, v[:, order], labels=[scores.series[i] for i in order]))))],
        ))  # fmt: skip
    return views


def prompt_views(scores: TokenScores, values: Scores, labels: list[str], per_label: int) -> list[TokenView]:
    span = float(values.abs().max())
    low, high = RAMPS["diverging"].light.low, RAMPS["diverging"].light.high

    def strip(label: str, tokens: list[str], v: Scores) -> Strip:
        html = colored_tokens(tokens, v[:, 0], min_value=-span, max_value=span, negative_color=low, positive_color=high)
        return Strip(label=label, html=closed(str(html)))

    views = []
    for text in first_per_label(scores.texts, labels, per_label):
        v, k = scores_of(text, values), text.marker
        assert k is not None, f"{text.id} has no reply marker"
        views.append(TokenView(
            title=f"{text.id} ({text.label})",
            subtitle=f"Projection at the last prompt token: {v[k - 1, 0]:.2f}. Reply mean: {v[k:, 0].mean():.2f}.",
            strips=[strip("prompt", text.tokens[:k], v[:k]), strip("reply", text.tokens[k:], v[k:])],
        ))  # fmt: skip
    return views
