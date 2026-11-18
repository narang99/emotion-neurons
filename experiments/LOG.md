# Experiment log

- One line per experiment, in order, failures included.
- Each has a tag at the commit that produced its results, and a detailed log in `experiments/`.
- Status: **kept** (in `main`, feeds the result), **failed** (branch and tag kept, not merged), **superseded**.

| tag | status | what | result | log |
|---|---|---|---|---|
| `exp/stories` | kept | 1680 Claude-written stories (12 emotions x 20 topics x 6, 240 neutral) | validator + audit pass | `PLAN.md` |
| `exp/stage2_l25` | kept | emotion vectors at layer 25 (mean minus the 12-emotion mean, neutral PCs removed) | held-out 12-way accuracy 0.81-0.84 | `stage2_l25.md` |
| `exp/stage3_l25` | kept | neurons behind each MLP write; 2M-token harvest; HDBSCAN modes named by Haiku | 0.9/0.9 dims-then-neurons keeps all 12288; pos-pole neurons 0.30 feeling-word modes vs controls 0.15 | `stage3_l25.md` |
| `exp/min_set_l25` | superseded | greedy minimum set per emotion write | cos 0.9 with 700-1050 neurons; replaced by sparse sets with a held-out stop | `top64_l25.md` |
| `exp/sparse_l25` | kept | sparse sets with held-out transfer; swaps, crossfit, shrunk targets; top-64 modes | 298-1233 neurons match the full MLP; no knee; hill-climbing gains <= 6.5% | `top64_l25.md` |
| `exp/sparse_l25` (round 1 of crossfit) | failed | crossfit with a cross-half Gram | indefinite Gram, transfer ~0; fixed by the halves' own norms | `top64_l25.md` |
| `exp/probe_kv_l25` | kept | top 64 on 60 adult texts; values through the unembedding | 3068 pride, 2862 laughter, 11109 racing heart read and write their concept; next-token agreement at chance | `probe_kv_l25.md` |
| `exp/probe_kv_l25` (mode assignment) | failed | assigning new hits to modes from 5 exemplars | too strict, 2% pass; needs saved full cluster centres | `probe_kv_l25.md` |
| `refactor/src-layout` | kept | refactor into a src/ library with thin scripts (`6c3f430`) | CPU and L4 regression identical: stage2 sums 1.6e-8, all 1440 labels, projections, harvest (bit-identical acts), modes, probe | commit message |
| `exp/journey_all` | kept | emotion vectors across all 36 layers: attention vs MLP shares, direction, decodability | held-out accuracy peaks at 0.87 (layer 22); layer-25 vector 66% attention / 34% MLP; last layer 40% / 60%; middle MLPs barely write along it | `journey_all.md` |
| `exp/sparse_layers` | kept | sparse sets of every layer's MLP emotion write (greedy, held-out stop) | cos 0.9 needs ~1200 neurons in layers 10-20, ~570 in 28-34, 38 at layer 35; k_match defined to +/-4% | `sparse_layers.md` |
| `exp/circuits_l25-round1` | failed | dense circuits for the layer-25 score, ranked by effect against zero | faithfulness -0.1 at 100-1000 neurons, 0.53 at 30000: ranked constant activations; fix: effect against the mean (round 2, pending) | `circuits_l25.md` |
| `exp/circuits_l25-round2` | failed | the same, ranked by effect against the mean | faithfulness -0.04 to -0.21 at 100-3000 neurons, 0.75 at 30000; joint mean ablation of 26 layers is beyond a first-order ranking | `circuits_l25.md` |
| `exp/neuron_circuits` | kept | local circuits of 3068 (pride) and 11109 (racing heart): exact input splits, readers, checks | shame / embarrassment / pride writers -> 3068 -> pride and posture readers; holding 3068 lowers " pride" 0.85 nats; 11109 -> pulse / unease readers, lowers " adrenal", " cortisol" | `neuron_circuits.md` |
