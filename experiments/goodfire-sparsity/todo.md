# Todo

- Refusal rate of the ablated model.
  - Generate replies to the harmful and harmless val prompts of `refusal_qwen1` under each ablation: all layers, the sweep, the random control.
  - Score them with the refusal check of `refusal_qwen1` (refusal phrases).
  - Plot the refusal rate of each category against the first ablated layer, both bands, in `report.html`.
  - Replaces the KL at the post-instruction positions as the measure of refusal.
- What the ablated model predicts where KL is high.
  - Per ablation, save the highest-KL positions: the context, the intact model's top tokens and the ablated model's top tokens, with probabilities.
  - Show them in `report.html`, for held-out text and for the refusal prompts.
