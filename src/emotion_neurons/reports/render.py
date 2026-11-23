"""Render report data into one self-contained HTML file with Jinja.

- Templates live in `reports/templates/`. Each report extends `base.html` and uses the macros in `macros.html`.
- Autoescape is on: text from data (story ids, prompts, replies) can never inject markup.
"""

from pathlib import Path

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from pydantic import BaseModel

ENVIRONMENT = Environment(
    loader=PackageLoader("emotion_neurons", "reports/templates"),
    autoescape=select_autoescape(["html"]),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render(template: str, report: BaseModel) -> str:
    return ENVIRONMENT.get_template(template).render(report=report)


def write_report(template: str, report: BaseModel, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(template, report))
    return path
