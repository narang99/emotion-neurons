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
