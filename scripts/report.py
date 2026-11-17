"""Stage 2c report: do the emotion vectors work on held-out stories? CPU, static HTML.

    python scripts/report.py --config configs/stage2_l25.yaml

Reads results/<run_id>/{projections*, checks.json, vectors_meta.json, logit_lens.json}.
Writes results/<run_id>/report/index.html and one token page per story label.

Projections are shown as z-scores against held-out neutral stories (tokens split_token+), per
vector set and emotion, so 0 = what a neutral story reads and 1 = one neutral standard deviation.
"""

import html
import json
import sys

import torch
from safetensors.torch import load_file

from emotion_neurons.config import EMOTIONS, SITES, load_run_config
from emotion_neurons.stories import LABELS

CV = "https://unpkg.com/circuitsvis@1.43.3/dist/cdn/esm.js"


def zscore(proj, stories, split):
    """Per column: subtract the neutral mean and divide by the neutral sd (held-out neutral, tokens split+)."""
    rows = torch.cat([proj[s["offset"] + split : s["offset"] + s["n_tokens"]] for s in stories if s["emotion"] == "neutral"])
    mu, sd = rows.mean(0), rows.std(0)
    return (proj - mu) / sd, mu, sd


def story_scores(z, stories, split):
    """Mean z over tokens split+, per story: [n_stories, 12]."""
    return torch.stack([z[s["offset"] + split : s["offset"] + s["n_tokens"]].mean(0) for s in stories])


def color(v, vmax):
    """Diverging: blue for negative, red for positive, white at 0."""
    t = max(-1.0, min(1.0, v / vmax)) if vmax else 0.0
    if t >= 0:
        return f"rgb(255,{int(255 * (1 - t))},{int(255 * (1 - t))})"
    return f"rgb({int(255 * (1 + t))},{int(255 * (1 + t))},255)"


def table(M, rows, cols, fmt="{:.2f}", vmax=None, bold_diag=False):
    vmax = vmax or max(abs(float(x)) for r in M for x in r) or 1.0
    out = ["<table class=m><tr><th></th>" + "".join(f"<th>{c[:5]}</th>" for c in cols) + "</tr>"]
    for i, (name, r) in enumerate(zip(rows, M)):
        cells = []
        for j, x in enumerate(r):
            style = f"background:{color(float(x), vmax)}" + (";font-weight:bold;outline:2px solid #333" if bold_diag and i == j else "")
            cells.append(f"<td style='{style}'>{fmt.format(float(x))}</td>")
        out.append(f"<tr><th>{name}</th>{''.join(cells)}</tr>")
    return "\n".join(out + ["</table>"])


def tokens_widget(div_id, tokens, values):
    payload = json.dumps({"tokens": tokens, "values": values, "labels": list(EMOTIONS)})
    return (
        f"<div id='{div_id}' style='margin:8px 0'></div>"
        f"<script type='module'>import {{ render, ColoredTokensMulti }} from '{CV}';"
        f"render('{div_id}', ColoredTokensMulti, {payload});</script>"
    )


STYLE = """<style>
body{font-family:system-ui,sans-serif;margin:16px;max-width:1200px;color:#222;background:#fff}
table.m{border-collapse:collapse;font-size:12px;margin:6px 0 18px}
table.m td,table.m th{border:1px solid #ddd;padding:3px 5px;text-align:right}
.grid{display:flex;flex-wrap:wrap;gap:24px}
h2{margin-top:32px;border-bottom:1px solid #ccc} .note{color:#555;font-size:14px;max-width:900px}
code{background:#f3f3f3;padding:1px 4px}
</style>"""


def main():
    cfg = load_run_config(sys.argv[1:])
    rd = cfg.result_dir
    proj = load_file(str(rd / "projections.safetensors"))
    pmeta = json.loads((rd / "projections_meta.json").read_text())
    checks = json.loads((rd / "checks.json").read_text())
    vmeta = json.loads((rd / "vectors_meta.json").read_text())
    lens = json.loads((rd / "logit_lens.json").read_text())
    stories, split = pmeta["stories"], pmeta["split_token"]
    labels = [s["emotion"] for s in stories]
    out_dir = rd / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = [f"<!doctype html><meta charset=utf-8><title>Emotion vectors check</title>{STYLE}",
             f"<h1>Emotion vectors, {html.escape(cfg.model)} layer {cfg.layer}, run <code>{cfg.run_id}</code></h1>",
             "<p class=note>Vectors are built from the train stories (100 per emotion, 200 neutral). Everything on this page "
             "except the nearest-vector labels uses only held-out stories (20 per emotion, 40 neutral). Projections are "
             f"z-scores against held-out neutral stories, tokens {split}+.</p>"]  # fmt: skip

    # 1. Held-out projection matrices, one per site and vector set.
    summary = []
    parts.append("<h2>1. Held-out projection matrix</h2><p class=note>Row: the story's label. Column: the vector. "
                 "Cell: mean z over the row's stories. The vectors work if each row peaks on its own column (outlined). "
                 "Accuracy: share of held-out emotional stories whose highest column is their own emotion.</p><div class=grid>")  # fmt: skip
    zs = {}
    for k in pmeta["keys"]:
        z, _, _ = zscore(proj[k].float(), stories, split)
        zs[k] = z
        S = story_scores(z, stories, split)
        M = [S[[i for i, l in enumerate(labels) if l == lab]].mean(0).tolist() for lab in LABELS]
        emo_idx = [i for i, l in enumerate(labels) if l != "neutral"]
        acc = sum(EMOTIONS[int(S[i].argmax())] == labels[i] for i in emo_idx) / len(emo_idx)
        diag = sum(M[e][e] for e in range(len(EMOTIONS))) / len(EMOTIONS)
        summary.append((k, acc, diag))
        parts.append(f"<div><b>{k}</b> &nbsp; acc {acc:.2f}, mean diagonal z {diag:.2f}" + table(M, LABELS, EMOTIONS, bold_diag=True) + "</div>")
    parts.append("</div>")
    parts.insert(3, "<h2>Summary</h2>" + table([[a, d] for _, a, d in summary], [k for k, _, _ in summary], ["acc", "diag z"], vmax=1e9))

    # 2. Nearest-vector labels on pooled means (all stories, train ones left out of their own mean).
    parts.append("<h2>2. Nearest-vector label (pooled means)</h2><p class=note>Each emotional story's mean activation "
                 "(tokens 50+) minus the grand mean, assigned to the closest vector (mean baseline, raw). Train stories are "
                 "left out of their own emotion's mean. Confusion over all 1440 stories; rows are labels, columns predictions.</p>")  # fmt: skip
    for site in SITES:
        c = checks["sites"][site]
        parts.append(f"<h3>{site}: train-LOO {c['acc_train_loo']:.3f}, held-out {c['acc_heldout']:.3f}</h3>")
        parts.append(table(c["confusion_all"], EMOTIONS, EMOTIONS, fmt="{:.0f}", vmax=120, bold_diag=True))
    fl = checks["flagged"]
    parts.append(f"<h3>Stories mislabelled at 2 or more sites: {len(fl)}</h3><table class=m><tr><th>id</th><th>label</th><th>held out</th><th>predicted by site</th></tr>")
    parts += [f"<tr><td>{f['id']}</td><td>{f['label']}</td><td>{f['heldout']}</td><td>{html.escape(json.dumps(f['pred_by_site']))}</td></tr>" for f in fl]
    parts.append("</table>")

    # 3. Stability and structure.
    parts.append("<h2>3. Split-half cosine</h2><p class=note>Vectors from topics 0-9 vs topics 10-19 (train stories), mean baseline, raw.</p>")
    parts.append(table([[checks["sites"][s]["split_half_cos"][e] for e in EMOTIONS] for s in SITES], SITES, EMOTIONS, vmax=1))
    parts.append("<h2>4. Emotion x emotion cosine (mean baseline, denoised)</h2><div class=grid>")
    for site in SITES:
        parts.append(f"<div><b>{site}</b>" + table(checks["sites"][site]["cos_mean_denoised"], EMOTIONS, EMOTIONS, vmax=1) + "</div>")
    parts.append("</div>")

    # 4. Logit lens.
    parts.append("<h2>5. Logit lens</h2><p class=note>Top tokens of W_U (gamma * unit vector), final RMSNorm gain folded in.</p>")
    for k, entry in lens.items():
        parts.append(f"<h3>{k}</h3><table class=m><tr><th>emotion</th><th>top</th><th>bottom</th></tr>")
        parts += [f"<tr><th>{e}</th><td style='text-align:left'>{html.escape(' | '.join(v['top'][:12]))}</td>"
                  f"<td style='text-align:left'>{html.escape(' | '.join(v['bottom'][:8]))}</td></tr>" for e, v in entry.items()]  # fmt: skip
        parts.append("</table>")

    # 5. Token pages.
    k = f"{cfg.report.story_site}/{cfg.report.story_vector}"
    parts.append(f"<h2>6. Token pages ({k})</h2><p>" + " · ".join(f"<a href='stories_{lab}.html'>{lab}</a>" for lab in LABELS) + "</p>")
    parts.append(f"<h2>Denoising</h2><pre>{html.escape(json.dumps(vmeta['denoise'], indent=1))}</pre>")
    (out_dir / "index.html").write_text("\n".join(parts))

    z = zs[k]
    for lab in LABELS:
        page = [f"<!doctype html><meta charset=utf-8><title>{lab} stories</title>{STYLE}",
                f"<h1>Held-out <i>{lab}</i> stories, coloured by {k}</h1><p class=note><a href='index.html'>back</a>. "
                f"Pick a vector in each widget. Values are z against neutral; tokens before {split} are excluded from the averages but shown here.</p>"]  # fmt: skip
        for j, s in enumerate(stories):
            if s["emotion"] != lab:
                continue
            vals = z[s["offset"] : s["offset"] + s["n_tokens"]]
            means = vals[split:].mean(0)
            ranked = ", ".join(f"{EMOTIONS[i]} {means[i]:.2f}" for i in means.argsort(descending=True)[:3].tolist())
            page.append(f"<h3>{s['id']} <small>(top: {ranked})</small></h3>")
            page.append(tokens_widget(f"s{j}", s["tokens"], [[round(x, 3) for x in r] for r in vals.tolist()]))
        (out_dir / f"stories_{lab}.html").write_text("\n".join(page))
    for k, a, d in summary:
        print(f"[report] {k}: held-out acc {a:.2f}, mean diagonal z {d:.2f}")
    print(f"[saved] {out_dir}/index.html + {len(LABELS)} story pages")


if __name__ == "__main__":
    main()
