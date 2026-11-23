# To do

Open questions, in no fixed order. Each becomes an experiment with a tag and a log when it starts.

## Refusal (Qwen-1 7B Chat, layer 17)

- **Neuron or direction: which detects harm better?**
  - Detectors: the layer-18 direction projection; a neuron past its general-text p99.9; a count of top-k neurons past p99.9.
  - Tests: AUC on the paper's val prompts; AUC on HH-RLHF `Assistant:` turns (harmless-base vs helpful-base);
    judged precision of each detector's top 0.1% of the harvest corpus.
  - Already known: at the reply token, 80% of harmful prompts put the top 40 neurons at p99+ of their general-text
    distribution (harmless 6%); n1958 is at p99.99+ on 61% of harmful prompts.
- **Harm neurons outside the set.** A neuron outside the refusal set does not build the direction.
  - If its poles are harm-related, what does it do instead? Does it write against refusal (negative share)?
  - Use such neurons to find prompts where refusal should fire but the direction does not, and the neuron does.
  - Starts from the all-neuron harvest and the harm-coverage judges.

## All-neuron harvest of layer 17 (code in progress, not run)

- Built: `emotion_neurons.harvest_all`, `scripts/harvest_all.py`, `configs/harvest_all_qwen1_l17.yaml`.
  - One model pass saves the MLP inputs (about 18 GB fp16 for 2.2M tokens); activations per chunk of 1024 neurons
    come from them; per-band hits and HDBSCAN as in `exp/harvest_bands`; gzipped store shards in runs/<run_id>/store/.
  - Smoke test on CPU (tiny random Qwen3, 40k tokens, 32 neurons): saved-input activations match the model pass
    within 0.013; shards written; pole shares sum to 100%.
  - Bug: after "[harvest-all] done" the process does not exit (likely the spawn worker pool). Fix before a GPU run.
- Not built yet:
  - a judge that reads the store with a prompt file and writes judgements_<tag>/, on all neurons or a sample;
  - a sharded report (index plus per-neuron cards) for 11008 neurons;
  - share of the refusal write for every neuron, signed, in the store or the report.
- Judge prompt: start from `data/prompts/refusal_categories.md` (constitution categories, hard constraints marked).
- Cost: both judges are $0.10 / $0.50 per M tokens on OpenRouter; a full pass is about 30M tokens, about $6 per model.
