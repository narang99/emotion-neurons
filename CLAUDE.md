# emotion-neurons

## Subagents: save the usage limit

- Spawn subagents only when the user asks.
- Bulk text generation (stories, prompts, datasets) and simple read/summarise tasks use `model: "haiku"`.
  - The first story batch ran 14 Opus agents (~1.25M tokens) and used ~95% of a 5-hour limit.
- Use Sonnet or Opus only for work that needs judgment: design, debugging, analysis.
- Parallel agents each get their own scratchpad subfolder; a shared one made agents overwrite each other.
- GPU work (generation with Qwen, extraction, harvesting) runs on Colab, not through Claude.

## Pushing

- `origin` uses SSH (`git@github.com:narang99/emotion-neurons.git`); a plain `git push` works.
- If a clone has an HTTPS remote, switch it: `git remote set-url origin git@github.com:narang99/emotion-neurons.git`.

## Prose

These hold in code, docs, READMEs, PRs and commits.

- One sentence says one thing. Do not compose several sentences into one.
- No em dashes. They invite exactly that kind of composed sentence.
- Use bullet points for most things.
  - Use nested bullets to continue a point, instead of a longer sentence.
- Be concise, but prefer simplicity over conciseness.

## Code

- Readability first, then conciseness.
- Small functions that do one thing, with names that carry the meaning.
  - A function is fine even when it is used once, if it makes the caller read better.
- Docstrings explain the why and how of non-trivial parts only. Never restate the signature.
- Strong typing everywhere.
  - `jaxtyping` for tensors (`Float[Tensor, "story d_model"]`), with named aliases for long types.
  - `einops` for tensor maths that has named axes.
  - Pydantic for everything else that is structured, including configs (`emotion_neurons.config`).
- Library code lives in `src/emotion_neurons/`. Scripts in `scripts/` stay thin: parse the config, call the library, save.
- How emotion vectors are built is isolated in `emotion_neurons.vectors`.
  - Downstream code reads `EmotionVectors`, never a recipe. A new procedure is a new entry in `RECIPES`.
- Tests in `tests/` check exact identities (hooks match a manual pass, quantiles match a sort).
- Formatting: `scripts/git-hooks/pre-commit` sorts imports and runs `ruff format` on staged files.
  - Install it once per clone: `ln -s ../../scripts/git-hooks/pre-commit .git/hooks/pre-commit`.

## Experiments and git

- `main` holds only work that feeds the final result. Refactors may go straight to `main`.
- New research work goes up as stacked PRs with `gh stack`, one branch and PR per experiment, for review.
- Every experiment gets a tag (`exp/<name>`) at the commit that produced its results, failed ones included.
- Each experiment is a folder `experiments/<name>/`, with no numbers in the name.
  - `README.md`: what was done, the scripts that were run, the results.
  - The YAML configs the scripts were run with.
  - `results/`: the committed outputs, reports included.
- Failures stay out of `main`. Their branch and tag are kept, and the README says why they failed.
- Commit messages carry the key numbers, so a later run can be checked against them.

## Running on Colab

- Use the `colab-experiments` skill: the dependency rules, the notebook loop and how to verify a run.
- Large outputs go to `runs/` (gitignored) and to S3; small ones to the experiment's `results/` (committed).
  - Public copies for collaborators: `scripts/fetch_runs.sh`.

## Archive

- The branch `archive` holds all work from before the cleanup of 2026-11-23. `archive.md` indexes it.
- Before you build a feature, read `archive.md`. If the feature exists there, start from that code.
- Archived features:
  - greedy neuron sets for an emotion write or a direction (`sparse/`);
  - harvests of what neurons fire on, per pole and percentile band, and an all-neuron harvest store;
  - modes (HDBSCAN clusters of hits), blind OpenRouter judges, themes, and the end-to-end modes pipeline;
  - the modes report, the logit lens, circuitsvis token strips, the direction prompt view;
  - theme removal from an emotion write;
  - about 40 tagged experiments (`exp/<name>`): layers, circuits, chains, causal sets, head reads, crossval.
